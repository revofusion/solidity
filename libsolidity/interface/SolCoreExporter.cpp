/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0

#include <libsolidity/interface/SolCoreExporter.h>

#include <libsolidity/ast/AST.h>
#include <libsolidity/ast/ASTAnnotations.h>
#include <libsolidity/interface/Version.h>

#include <liblangutil/Token.h>

#include <libyul/AST.h>
#include <libyul/Utilities.h>

#include <libsolutil/Visitor.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace solidity;
using namespace solidity::frontend;
using namespace solidity::langutil;
using namespace std::string_literals;

namespace
{

struct UnsupportedSolCore: std::runtime_error
{
	using std::runtime_error::runtime_error;
};

// --- Namespaced storage (ERC-7201) support ---
//
// A namespaced storage getter is a function like:
//   function _getTokenStorage() private pure returns (TokenStorage storage $) {
//       assembly { $.slot := TOKEN_STORAGE_LOCATION }
//   }
//
// We detect these, flatten the struct fields into the main Storage type,
// and rewrite accesses through the storage pointer ($) into direct
// storage_get / storage_set / storage_map_get / storage_map_set operations.

/// Information about a detected namespaced storage getter.
struct NamespacedStorageGetter
{
	FunctionDefinition const* function;        ///< The getter function AST node
	StructDefinition const* structDef;         ///< The struct type it returns
	std::string prefix;                        ///< Field name prefix (e.g. "token_")
	std::string fieldName;                     ///< Synthetic Storage field (e.g. "token")
};

/// Thread-local map: local variable name → field prefix for namespaced storage aliases.
/// Populated when we encounter `TokenStorage storage $ = _getTokenStorage();`
/// and consulted when we encounter `$.field` or `$.mapping[key]`.
static thread_local std::map<std::string, std::string> namespacedStorageAliases;

/// Thread-local map from namespaced getter definitions to their field prefix,
/// so we can resolve overloaded getters precisely during export.
static thread_local std::map<FunctionDefinition const*, std::string> namespacedGetterPrefixes;

/// Thread-local set of struct member names that are mappings, keyed by
/// prefixed field name → true if the field is a mapping type.
static thread_local std::set<std::string> namespacedMappingFields;
static thread_local std::map<FunctionDefinition const*, std::string> exportedFunctionNames;
static thread_local CompilerStack const* activeCompilerStack = nullptr;
static thread_local ContractDefinition const* activeExportContract = nullptr;

// --- `unchecked { }` block signal (SOLCORE_MATH_BUG_CLASSES_PLAN.md §3.5/Task 11) ---
//
// Before this, `unchecked { ... }` carried zero signal through the exporter:
// every arithmetic op lowered to the same JSON regardless of source-level
// `unchecked`, so the OCaml frontend had no way to distinguish a checked add
// from a wrapping one. This is a plain ambient (thread-local) flag rather
// than a threaded parameter because `exportExpr`/`exportStmt` are mutually
// recursive free functions called from dozens of sites; threading a new
// parameter through all of them would be a much larger, higher-blast-radius
// change than this file's existing `thread_local` ambient-context idiom
// (`namespacedStorageAliases` et al., above) for exactly this kind of
// "ambient fact about where we are in the AST" state.
static thread_local bool inUncheckedBlock = false;

/// RAII guard: sets `inUncheckedBlock` for the duration of walking one
/// `Block`'s statements, restoring the previous value on scope exit (so
/// nested checked blocks inside an unchecked one, and vice versa, are
/// handled correctly — Solidity's `unchecked` does not nest by inheriting
/// the enclosing block's mode, each `Block` carries its own flag).
struct UncheckedBlockGuard
{
	explicit UncheckedBlockGuard(bool _unchecked):
		m_previous(inUncheckedBlock)
	{
		inUncheckedBlock = _unchecked;
	}
	~UncheckedBlockGuard() { inUncheckedBlock = m_previous; }
	UncheckedBlockGuard(UncheckedBlockGuard const&) = delete;
	UncheckedBlockGuard& operator=(UncheckedBlockGuard const&) = delete;
private:
	bool m_previous;
};

/// Tag a just-built arithmetic-op JSON node with `"unchecked": true` when it
/// was produced while walking an `unchecked { }` block, so the OCaml
/// frontend can choose the wrapping (`yul_*`) model instead of the checked
/// (`u256_*_checked`) one at that site. Additive-only: absent (the default
/// for every pre-existing corpus JSON file, and every checked-context site)
/// means exactly what it always meant — checked arithmetic — so this cannot
/// change behavior for any already-exported JSON.
void markUncheckedContext(Json& _result)
{
	if (inUncheckedBlock)
		_result["unchecked"] = true;
}

// --- Declared-width signal for narrow-integer arithmetic (SolCore audit:
// narrow-width arithmetic soundness gap) ---
//
// Solidity >=0.8 bounds checked arithmetic at the DECLARED type's width
// (`uint8 + uint8` reverts past 255), and `unchecked { }` arithmetic wraps
// at that same declared width — never at 2^256. Before this tag existed,
// `u256_add`/`u256_sub`/`u256_mul` JSON nodes carried no width at all, so
// the OCaml frontend could only model every add/sub/mul at the full
// 256-bit word: a `uint8` sum of 250+15 was modeled as succeeding with 265
// instead of reverting (checked) or wrapping to 9 (unchecked) — silently
// masking narrow-overflow bypasses of downstream `require` gates.
//
// `tagNarrowArithWidth` records the operation's static integer width as
// `"bits": N` (8..248) on the just-built arithmetic node whenever the
// operation is typed at a narrow UNSIGNED integer type. The right type to
// consult is the OPERATION's type — `commonType` for a `BinaryOperation`,
// the mutated/assigned expression's own type for `++`/`--`/`+=`-family
// sites — which the type checker has already resolved through implicit
// conversions, so `uint256(a) + uint256(b)` (256-bit arithmetic on widened
// narrow operands) correctly stays untagged while `a + b` on two `uint8`s
// is tagged 8. Additive-only: absent means what it always meant (the full
// 256-bit word), so pre-existing corpus JSON keeps its exact old meaning;
// the `featureFlags.arithWidths` marker lets consumers detect artifacts
// that predate this signal. Narrow SIGNED types are deliberately NOT
// tagged: signed arithmetic has no faithful SolCore lowering yet (audit
// bug #5's fail-closed domain), and a width tag alone would not repair the
// missing two's-complement value model, so signed sites keep the
// status-quo lowering unchanged.
void tagNarrowArithWidth(Json& _result, Type const* _operationType)
{
	if (!_operationType)
		return;
	auto const* intType = dynamic_cast<IntegerType const*>(_operationType);
	if (intType && !intType->isSigned() && intType->numBits() < 256)
		_result["bits"] = static_cast<int>(intType->numBits());
}

std::string exportedFunctionName(FunctionDefinition const& _function);

template <class F>
void forEachEnumDefinition(SourceUnit const& _unit, F&& _f)
{
	for (auto const& node : _unit.nodes())
	{
		if (auto const* enumDef = dynamic_cast<EnumDefinition const*>(node.get()))
			_f(*enumDef);
		else if (auto const* contractNode = dynamic_cast<ContractDefinition const*>(node.get()))
			for (auto const& subNode : contractNode->subNodes())
				if (auto const* enumDef = dynamic_cast<EnumDefinition const*>(subNode.get()))
					_f(*enumDef);
	}
}

std::string enumDefinitionIdentity(EnumDefinition const& _enumDef)
{
	return _enumDef.sourceUnitName() + ":" + _enumDef.name() + "#" + std::to_string(_enumDef.id());
}

bool enumNameNeedsQualification(EnumDefinition const& _enumDef)
{
	if (!activeCompilerStack)
		return false;
	bool seenDifferentDefinition = false;
	auto const ownIdentity = enumDefinitionIdentity(_enumDef);
	for (auto const& sourceName: activeCompilerStack->sourceNames())
	{
		try
		{
			forEachEnumDefinition(
				activeCompilerStack->ast(sourceName),
				[&](EnumDefinition const& candidate) {
					if (
						candidate.name() == _enumDef.name() &&
						enumDefinitionIdentity(candidate) != ownIdentity
					)
						seenDifferentDefinition = true;
				});
		}
		catch (...)
		{
		}
	}
	return seenDifferentDefinition;
}

std::string exportedEnumName(EnumDefinition const& _enumDef)
{
	if (enumNameNeedsQualification(_enumDef))
		return enumDefinitionIdentity(_enumDef);
	return _enumDef.name();
}

/// Derive a field prefix from a struct name.
/// "TokenStorage" → "token_", "MyDataStorage" → "myData_"
std::string derivePrefix(std::string const& _structName)
{
	std::string base = _structName;
	// Strip "Storage" suffix if present
	if (base.size() > 7 && base.substr(base.size() - 7) == "Storage")
		base = base.substr(0, base.size() - 7);
	// Lowercase the first character
	if (!base.empty())
		base[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(base[0])));
	return base + "_";
}

std::string deriveSubStorageFieldName(std::string const& _structName)
{
	std::string base = _structName;
	if (base.size() > 7 && base.substr(base.size() - 7) == "Storage")
		base = base.substr(0, base.size() - 7);
	for (char& ch: base)
		ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	return base;
}

std::string const* namespacedGetterPrefix(Identifier const& _callee)
{
	auto const* funcDef =
		dynamic_cast<FunctionDefinition const*>(_callee.annotation().referencedDeclaration);
	if (!funcDef)
		return nullptr;
	auto it = namespacedGetterPrefixes.find(funcDef);
	if (it == namespacedGetterPrefixes.end())
		return nullptr;
	return &it->second;
}

/// Check if a function is a namespaced storage getter.
/// Criteria:
///   1. Returns exactly one parameter with Location::Storage
///   2. Return type references a struct
///   3. Body is a single inline assembly block or storage-slot local plus inline assembly
bool isNamespacedStorageGetter(
	FunctionDefinition const& _function,
	StructDefinition const** _outStructDef)
{
	// Must be an ordinary function (not constructor, receive, or fallback).
	// Checking this first avoids calling visibility() on a constructor,
	// which triggers solAssert(!isConstructor()) inside defaultVisibility().
	if (!_function.isOrdinary())
		return false;

	// Must be private or internal
	if (_function.visibility() == Visibility::Public || _function.visibility() == Visibility::External)
		return false;

	// Must return exactly one parameter with storage location
	if (_function.returnParameters().size() != 1)
		return false;
	auto const& retParam = *_function.returnParameters().front();
	if (retParam.referenceLocation() != VariableDeclaration::Location::Storage)
		return false;

	// Return type must reference a struct
	if (auto const* userDefined = dynamic_cast<UserDefinedTypeName const*>(&retParam.typeName()))
	{
		Declaration const* referencedDecl = userDefined->pathNode().annotation().referencedDeclaration;
		if (auto const* structDef = dynamic_cast<StructDefinition const*>(referencedDecl))
		{
			*_outStructDef = structDef;
		}
		else
			return false;
	}
	else
		return false;

	// Body should contain an inline assembly block that sets $.slot.
	// Pattern 1 (direct): { assembly { $.slot := CONSTANT } }
	// Pattern 2 (indirect): { bytes32 slot = fn(); assembly { $.slot := slot } }
	if (!_function.isImplemented())
		return false;

	Block const& body = _function.body();
	size_t stmtCount = body.statements().size();

	if (stmtCount == 1)
	{
		// Pattern 1: single inline assembly block
		if (!dynamic_cast<InlineAssembly const*>(body.statements().front().get()))
			return false;
	}
	else if (stmtCount == 2)
	{
		// Pattern 2: variable declaration followed by inline assembly
		if (!dynamic_cast<VariableDeclarationStatement const*>(body.statements()[0].get()))
			return false;
		if (!dynamic_cast<InlineAssembly const*>(body.statements()[1].get()))
			return false;
	}
	else
		return false;

	return true;
}

/// RAII helper to set up and tear down namespaced storage aliases for a function scope.
struct NamespacedStorageScope
{
	std::map<std::string, std::string> savedAliases;
	std::map<FunctionDefinition const*, std::string> savedPrefixes;
	std::set<std::string> savedMappingFields;

	NamespacedStorageScope()
	{
		savedAliases = namespacedStorageAliases;
		savedPrefixes = namespacedGetterPrefixes;
		savedMappingFields = namespacedMappingFields;
	}
	~NamespacedStorageScope()
	{
		namespacedStorageAliases = savedAliases;
		namespacedGetterPrefixes = savedPrefixes;
		namespacedMappingFields = savedMappingFields;
	}
};

// --- End namespaced storage infrastructure ---

Json exporterMetadata(std::string const& _contractName)
{
	Json metadata = Json::object();
	metadata["schemaVersion"] = "0.1.0";
	metadata["compilerVersion"] = VersionString;
	metadata["contract"] = _contractName;
	if (VersionCompactBytes.size() >= 2)
		metadata["exporterFamily"] =
			"solcore-solidity-"s +
			std::to_string(static_cast<unsigned>(VersionCompactBytes[0])) + "." +
			std::to_string(static_cast<unsigned>(VersionCompactBytes[1]));
	else
		metadata["exporterFamily"] = "solcore-solidity-unknown";
	return metadata;
}

Json featureFlags()
{
	Json flags = Json::object();
	flags["arrays"] = true;
	flags["events"] = true;
	flags["constructors"] = true;
	flags["externalCalls"] = true;
	flags["multipleReturns"] = false;
	flags["structs"] = false;
	flags["enums"] = false;
	flags["inheritance"] = false;
	flags["modifiers"] = true;
	flags["inlineAssembly"] = true;
	flags["fixedBytes"] = true;
	flags["smallUints"] = true;
	flags["signedInts"] = true;
	// Narrow-unsigned arithmetic nodes carry their declared width as
	// `"bits": N` (see tagNarrowArithWidth). Artifacts without this flag
	// predate the signal: their untagged add/sub/mul nodes are AMBIGUOUS
	// between genuine 256-bit arithmetic and mis-modeled narrow arithmetic,
	// and consumers that need the distinction must treat them as stale.
	flags["arithWidths"] = true;
	return flags;
}

Json unsupportedExport(std::string const& _contractName, std::string const& _reason)
{
	Json result = Json::object();
	result["unsupported"] = true;
	result["reason"] = _reason;
	Json metadata = exporterMetadata(_contractName);
	for (auto const& [key, value]: metadata.items())
		result[key] = value;
	return result;
}

Json sourceLocation(SourceLocation const& _location)
{
	Json result = Json::object();
	if (_location.sourceName)
		result["file"] = *_location.sourceName;
	result["start"] = _location.start;
	result["end"] = _location.end;
	return result;
}

Json jsonStringArray(std::vector<std::string> const& _values)
{
	Json result = Json::array();
	for (auto const& value: _values)
		result.emplace_back(value);
	return result;
}

std::string exportedContractId(ContractDefinition const& _contract)
{
	return _contract.fullyQualifiedName();
}

void addInternalLibraryCallContractId(Json& _result, FunctionDefinition const& _function)
{
	auto const* library = dynamic_cast<ContractDefinition const*>(_function.scope());
	if (library && library->isLibrary())
		_result["contractId"] = exportedContractId(*library);
}

std::string solcoreDeployedCodeSize(ContractDefinition const& _contract)
{
	// SolCore export is intentionally available after semantic analysis and
	// must not force bytecode generation. Constructor semantics only need the
	// source-backed fact that deployable contracts have non-empty runtime code
	// after construction; exact byte size belongs to the EVM bytecode artifact.
	if (_contract.isInterface() || _contract.abstract())
		return "0";
	return "1";
}

std::optional<ContractDefinition const*> uniqueInterfaceImplementation(
	CompilerStack const& _compilerStack,
	ContractDefinition const& _interfaceContract
)
{
	if (!_interfaceContract.isInterface())
		return std::nullopt;

	auto const requiredFunctions = _interfaceContract.interfaceFunctions();
	if (requiredFunctions.empty())
		return std::nullopt;

	std::vector<ContractDefinition const*> candidates;
	for (std::string const& candidateName: _compilerStack.contractNames())
	{
		ContractDefinition const& candidate = _compilerStack.contractDefinition(candidateName);
		if (candidate.fullyQualifiedName() == _interfaceContract.fullyQualifiedName())
			continue;
		if (candidate.isInterface() || candidate.isLibrary() || candidate.abstract())
			continue;

		auto const candidateFunctions = candidate.interfaceFunctions();
		bool coversInterface = true;
		for (auto const& [selector, functionType]: requiredFunctions)
		{
			(void)functionType;
			if (!candidateFunctions.count(selector))
			{
				coversInterface = false;
				break;
			}
		}
		if (coversInterface)
			candidates.emplace_back(&candidate);
	}

	std::sort(
		candidates.begin(),
		candidates.end(),
		[](ContractDefinition const* lhs, ContractDefinition const* rhs)
		{
			return lhs->fullyQualifiedName() < rhs->fullyQualifiedName();
		}
	);
	candidates.erase(
		std::unique(
			candidates.begin(),
			candidates.end(),
			[](ContractDefinition const* lhs, ContractDefinition const* rhs)
			{
				return lhs->fullyQualifiedName() == rhs->fullyQualifiedName();
			}
		),
		candidates.end()
	);

	if (candidates.size() != 1)
		return std::nullopt;
	return candidates.front();
}

std::optional<Json> exportSimpleType(Type const& _type)
{
	switch (_type.category())
	{
	case Type::Category::Address:
	case Type::Category::Contract:
		return Json("address");
	case Type::Category::Bool:
		return Json("bool");
	case Type::Category::Integer:
	case Type::Category::FixedBytes:
		return Json("u256");
	default:
		return std::nullopt;
	}
}

std::optional<Json> exportStorageLayoutLeafType(Json const& _types, std::string const& _typeId)
{
	if (!_types.contains(_typeId))
		return std::nullopt;
	Json const& typeInfo = _types.at(_typeId);
	std::string const label = typeInfo.value("label", "");
	if (label == "address")
		return Json("address");
	if (label == "bool")
		return Json("bool");
	if (
		label == "uint256" ||
		label == "int256" ||
		(label.size() > 4 && (label.substr(0, 4) == "uint" || label.substr(0, 3) == "int")) ||
		(label.size() > 5 && label.substr(0, 5) == "bytes")
	)
		return Json("u256");
	return std::nullopt;
}

bool exportStorageLayoutType(
	Json const& _types,
	std::string const& _typeId,
	Json& _valueType,
	Json& _keys
)
{
	if (!_types.contains(_typeId))
		return false;
	Json const& typeInfo = _types.at(_typeId);
	if (typeInfo.value("encoding", "") == "mapping")
	{
		std::string const keyId = typeInfo.value("key", "");
		std::string const valueId = typeInfo.value("value", "");
		auto keyType = exportStorageLayoutLeafType(_types, keyId);
		if (!keyType.has_value())
			return false;
		_keys.emplace_back(*keyType);
		return exportStorageLayoutType(_types, valueId, _valueType, _keys);
	}
	auto valueType = exportStorageLayoutLeafType(_types, _typeId);
	if (!valueType.has_value())
		return false;
	_valueType = *valueType;
	return true;
}

std::string externalMutabilityString(StateMutability _mutability)
{
	return _mutability <= StateMutability::View ? "view" : "stateful";
}

std::optional<std::string> lookupStorageSlot(
	CompilerStack const& _compilerStack,
	ContractDefinition const& _contract,
	std::string const& _fieldName
)
{
	try
	{
		Json const& layout = _compilerStack.storageLayout(_contract.fullyQualifiedName());
		for (auto const& entry: layout.at("storage"))
			if (entry.value("label", "") == _fieldName)
				return entry.value("slot", "0");
	}
	catch (...)
	{
	}
	return std::nullopt;
}

std::optional<int> functionParameterIndex(
	FunctionDefinition const& _function,
	Declaration const* _declaration
)
{
	if (!_declaration)
		return std::nullopt;
	for (size_t i = 0; i < _function.parameters().size(); ++i)
		if (_function.parameters()[i].get() == _declaration)
			return static_cast<int>(i);
	return std::nullopt;
}

bool collectStorageGetterAccess(
	Expression const& _expr,
	FunctionDefinition const& _function,
	VariableDeclaration const*& _stateVar,
	std::vector<int>& _keyArgOrder
)
{
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
	{
		auto const* variable =
			dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (variable && variable->isStateVariable())
		{
			_stateVar = variable;
			return true;
		}
		return false;
	}
	if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr))
	{
		if (!indexAccess->indexExpression())
			return false;
		if (!collectStorageGetterAccess(
				indexAccess->baseExpression(),
				_function,
				_stateVar,
				_keyArgOrder
			))
			return false;
		auto const* keyIdentifier =
			dynamic_cast<Identifier const*>(indexAccess->indexExpression());
		if (!keyIdentifier)
			return false;
		auto paramIndex = functionParameterIndex(
			_function,
			keyIdentifier->annotation().referencedDeclaration
		);
		if (!paramIndex.has_value())
			return false;
		_keyArgOrder.emplace_back(*paramIndex);
		return true;
	}
	return false;
}

std::optional<Json> exportStorageGetterResolution(
	CompilerStack const& _compilerStack,
	ContractDefinition const& _contract,
	VariableDeclaration const& _stateVariable,
	std::vector<int> const& _keyArgOrder
)
{
	if (!_stateVariable.isStateVariable())
		return std::nullopt;
	auto slot = lookupStorageSlot(_compilerStack, _contract, _stateVariable.name());
	if (!slot.has_value())
		return std::nullopt;
	Type const* currentType = _stateVariable.annotation().type;
	if (!currentType)
		currentType = _stateVariable.type();
	if (!currentType)
		return std::nullopt;
	for (size_t i = 0; i < _keyArgOrder.size(); ++i)
	{
		auto const* mappingType = dynamic_cast<MappingType const*>(currentType);
		if (!mappingType)
			return std::nullopt;
		currentType = mappingType->valueType();
	}
	if (currentType->category() == Type::Category::Mapping)
		return std::nullopt;
	auto returnType = exportSimpleType(*currentType);
	if (!returnType.has_value())
		return std::nullopt;

	Json resolution = Json::object();
	resolution["kind"] = "storage_getter";
	resolution["field"] = _stateVariable.name();
	resolution["slot"] = *slot;
	Json keyArgOrder = Json::array();
	for (int index: _keyArgOrder)
		keyArgOrder.emplace_back(index);
	resolution["keyArgOrder"] = std::move(keyArgOrder);
	resolution["returnType"] = *returnType;
	return resolution;
}

std::optional<Json> exportStorageGetterResolution(
	CompilerStack const& _compilerStack,
	ContractDefinition const& _contract,
	FunctionDefinition const& _function
)
{
	if (!_function.isImplemented())
		return std::nullopt;
	if (_function.body().statements().size() != 1)
		return std::nullopt;
	auto const* returnStmt =
		dynamic_cast<Return const*>(_function.body().statements().front().get());
	if (!returnStmt || !returnStmt->expression())
		return std::nullopt;
	VariableDeclaration const* stateVar = nullptr;
	std::vector<int> keyArgOrder;
	if (!collectStorageGetterAccess(
			*returnStmt->expression(),
			_function,
			stateVar,
			keyArgOrder
		))
		return std::nullopt;
	if (!stateVar)
		return std::nullopt;
	return exportStorageGetterResolution(
		_compilerStack,
		_contract,
		*stateVar,
		keyArgOrder
	);
}

Json exportForeignMethodSummary(FunctionTypePointer const& _funType)
{
	Json method = Json::object();
	method["name"] = _funType->declaration().name();
	method["signature"] = _funType->externalSignature();
	method["selector"] = _funType->externalIdentifierHex();
	method["mutability"] = externalMutabilityString(_funType->stateMutability());
	TypePointers const& returns = _funType->returnParameterTypes();
	if (returns.size() == 1)
	{
		if (auto returnType = exportSimpleType(*returns.front()))
			method["return"] = *returnType;
	}
	else if (returns.size() > 1)
	{
		// Multi-return: export as a tuple type
		Json elements = Json::array();
		bool allResolved = true;
		for (auto const& ret : returns)
		{
			if (auto retType = exportSimpleType(*ret))
				elements.emplace_back(*retType);
			else
			{
				// Fall back to u256 for unsupported types
				elements.emplace_back(Json("u256"));
			}
		}
		if (allResolved)
		{
			Json tupleType = Json::object();
			tupleType["kind"] = "tuple";
			tupleType["elements"] = elements;
			method["return"] = tupleType;
		}
	}
	return method;
}

Json exportRevertPayload(FunctionCall const& _call);

Json exportDispatchEntry(FunctionTypePointer const& _funType, FunctionDefinition const* _functionDef = nullptr)
{
	Json entry = Json::object();
	entry["abiFunction"] = _funType->declaration().name();
	entry["function"] = _functionDef ? exportedFunctionName(*_functionDef) : _funType->declaration().name();
	entry["signature"] = _funType->externalSignature();
	entry["selector"] = _funType->externalIdentifierHex();
	if (_functionDef && _functionDef->isImplemented())
	{
		struct RevertPayloadCollector: ASTConstVisitor
		{
			std::vector<Json> payloads;
			std::set<std::string> seen;

			void pushPayload(Json _payload)
			{
				std::string key = _payload.dump();
				if (seen.insert(key).second)
					payloads.emplace_back(std::move(_payload));
			}

			bool visit(FunctionCall const& _call) override
			{
				Declaration const* calleeDecl = nullptr;
				if (auto const* callee = dynamic_cast<Identifier const*>(&_call.expression()))
					calleeDecl = callee->annotation().referencedDeclaration;
				else if (auto const* callee = dynamic_cast<MemberAccess const*>(&_call.expression()))
					calleeDecl = callee->annotation().referencedDeclaration;
				if (auto const* callee = dynamic_cast<Identifier const*>(&_call.expression()))
				{
					if (callee->name() == "require" || callee->name() == "revert")
						pushPayload(exportRevertPayload(_call));
					else if (callee->name() == "assert")
						pushPayload(Json::object());
				}
				if (dynamic_cast<ErrorDefinition const*>(calleeDecl))
					pushPayload(exportRevertPayload(_call));
				return true;
			}

			bool visit(RevertStatement const& _stmt) override
			{
				pushPayload(exportRevertPayload(_stmt.errorCall()));
				return true;
			}
		};

		RevertPayloadCollector collector;
		_functionDef->body().accept(collector);
		if (!collector.payloads.empty())
		{
			entry["revertPayloads"] = Json::array();
			for (auto& payload: collector.payloads)
				entry["revertPayloads"].emplace_back(std::move(payload));
		}
	}
	return entry;
}

Json exportForeignContractSummary(CompilerStack const& _compilerStack, std::string const& _contractName)
{
	ContractDefinition const& contract = _compilerStack.contractDefinition(_contractName);
	Json summary = Json::object();
	summary["id"] = exportedContractId(contract);
	summary["name"] = contract.name();

	Json storage = Json::array();
	try
	{
		Json const& layout = _compilerStack.storageLayout(_contractName);
		Json const& types = layout.at("types");
		for (auto const& entry: layout.at("storage"))
		{
			Json field = Json::object();
			field["name"] = entry.value("label", "");
			field["slot"] = entry.value("slot", "0");
			Json valueType = Json();
			Json keys = Json::array();
			if (exportStorageLayoutType(types, entry.value("type", ""), valueType, keys))
			{
				field["valueType"] = valueType;
				field["keys"] = keys;
				storage.emplace_back(std::move(field));
			}
		}
	}
	catch (...)
	{
	}
	summary["storage_layout"] = storage;
	summary["storage"] = storage;

	Json methods = Json::array();
	Json dispatchEntries = Json::array();
	try
	{
		for (auto const& [selector, functionType]: contract.interfaceFunctions())
		{
			(void)selector;
			if (!functionType)
				continue;
			methods.emplace_back(exportForeignMethodSummary(functionType));
			dispatchEntries.emplace_back(
				exportDispatchEntry(
					functionType,
					dynamic_cast<FunctionDefinition const*>(&functionType->declaration())
				)
			);
		}
	}
	catch (...)
	{
	}
	summary["methods"] = methods;
	summary["dispatch_entries"] = dispatchEntries;
	return summary;
}

std::optional<Json> exportKnownExternalTarget(
	CompilerStack const& _compilerStack,
	MemberAccess const& _memberAccess
)
{
	Type const* baseType = _memberAccess.expression().annotation().type;
	if (!baseType || baseType->category() != Type::Category::Contract)
		return std::nullopt;
	auto const* contractType = dynamic_cast<ContractType const*>(baseType);
	if (!contractType)
		return std::nullopt;

	Json target = Json::object();
	ContractDefinition const* targetContract = &contractType->contractDefinition();
	if (auto const* baseIdentifier = dynamic_cast<Identifier const*>(&_memberAccess.expression()))
		if (baseIdentifier->name() == "this" && activeExportContract)
			targetContract = activeExportContract;
	target["contractId"] = exportedContractId(*targetContract);
	if (targetContract == &contractType->contractDefinition())
		if (auto implementation = uniqueInterfaceImplementation(_compilerStack, *targetContract))
	{
		target["declaredContractId"] = exportedContractId(*targetContract);
		target["contractId"] = exportedContractId(**implementation);
		target["producerResolution"] = "unique_interface_implementation";
	}
	target["function"] = _memberAccess.memberName();
	target["resolutionKind"] = "cross_contract";

	if (auto const* functionType = dynamic_cast<FunctionType const*>(_memberAccess.annotation().type))
	{
		target["signature"] = functionType->externalSignature();
		target["selector"] = functionType->externalIdentifierHex();
		target["mutability"] = externalMutabilityString(functionType->stateMutability());
	}
	if (auto const* variableDef =
			dynamic_cast<VariableDeclaration const*>(_memberAccess.annotation().referencedDeclaration))
	{
		target["selector"] = variableDef->externalIdentifierHex();
		if (auto resolution =
				exportStorageGetterResolution(_compilerStack, *targetContract, *variableDef, {}))
		{
			target["resolutionKind"] = "storage_getter";
			target["resolution"] = *resolution;
		}
	}
	else if (auto const* functionDef =
				dynamic_cast<FunctionDefinition const*>(_memberAccess.annotation().referencedDeclaration))
	{
		FunctionType functionType(*functionDef);
		if (FunctionType const* iface = functionType.interfaceFunctionType())
		{
			if (!target.contains("signature"))
				target["signature"] = iface->externalSignature();
			if (!target.contains("selector"))
				target["selector"] = iface->externalIdentifierHex();
			if (!target.contains("mutability"))
				target["mutability"] = externalMutabilityString(iface->stateMutability());
		}
		else if (!target.contains("mutability"))
			target["mutability"] = externalMutabilityString(functionDef->stateMutability());
		if (!target.contains("selector"))
			target["selector"] = functionDef->externalIdentifierHex();
		if (auto resolution =
				exportStorageGetterResolution(_compilerStack, *targetContract, *functionDef))
		{
			target["resolutionKind"] = "storage_getter";
			target["resolution"] = *resolution;
		}
		if (auto const* funType = functionDef->functionType(false))
		{
			Json dispatchEntry = exportDispatchEntry(
				funType,
				functionDef
			);
			if (dispatchEntry.contains("revertPayloads"))
				target["revertPayloads"] = dispatchEntry["revertPayloads"];
		}
	}
	else if (!target.contains("mutability"))
		target["mutability"] = "stateful";
	return target;
}

char const* lowLevelCallKindString(std::string const& _memberName)
{
	return _memberName == "delegatecall" ? "delegatecall" : "call";
}

Json exportTypeName(TypeName const& _typeName);

Json exportTypeName(TypeName const& _typeName)
{
	if (auto const* elementary = dynamic_cast<ElementaryTypeName const*>(&_typeName))
	{
		switch (elementary->typeName().token())
		{
		case Token::Bool:
			return Json("bool");
		case Token::Address:
			return Json("address");
		case Token::UInt:
			return Json("u256");
		case Token::UIntM:
		{
			unsigned bits = elementary->typeName().firstNumber();
			switch (bits)
			{
			case 8:   return Json("u8");
			case 16:  return Json("u16");
			case 32:  return Json("u32");
			case 64:  return Json("u64");
			case 128: return Json("u128");
			case 256: return Json("u256");
			default:
			{
				Json result = Json::object();
				result["kind"] = "narrowed";
				result["base"] = "u256";
				result["bits"] = bits;
				return result;
			}
			}
		}
		case Token::Int:
			return Json("i256");
		case Token::IntM:
		{
			unsigned bits = elementary->typeName().firstNumber();
			switch (bits)
			{
			case 8:   return Json("i8");
			case 16:  return Json("i16");
			case 32:  return Json("i32");
			case 64:  return Json("i64");
			case 128: return Json("i128");
			case 256: return Json("i256");
			default:
			{
				Json result = Json::object();
				result["kind"] = "narrowed";
				result["base"] = "i256";
				result["bits"] = bits;
				return result;
			}
			}
		}
		case Token::BytesM:
		{
			unsigned bytes = elementary->typeName().firstNumber();
			return Json("bytes" + std::to_string(bytes));
		}
		case Token::Bytes:
		{
			// Dynamic bytes type — treat as an array of u8
			Json result = Json::object();
			result["kind"] = "array";
			result["element"] = Json("u8");
			return result;
		}
		case Token::String:
		{
			// String type — treat as an array of u8 at EVM level
			Json result = Json::object();
			result["kind"] = "array";
			result["element"] = Json("u8");
			return result;
		}
		default:
			break;
		}
		throw UnsupportedSolCore("Unsupported elementary type: " + elementary->typeName().toString());
	}
	if (auto const* mapping = dynamic_cast<Mapping const*>(&_typeName))
	{
		Json result = Json::object();
		result["kind"] = "mapping";
		result["key"] = exportTypeName(mapping->keyType());
		result["value"] = exportTypeName(mapping->valueType());
		return result;
	}
	if (auto const* arrayType = dynamic_cast<ArrayTypeName const*>(&_typeName))
	{
		Json result = Json::object();
		Json element = exportTypeName(arrayType->baseType());
		if (arrayType->length())
		{
			// Fixed-size array: uint256[10] or uint256[2**10 + 3]
			// First try to get the size from the literal expression
			bool sizeResolved = false;
			if (auto const* literal = dynamic_cast<Literal const*>(arrayType->length()))
			{
				try
				{
					result["kind"] = "fixed_array";
					result["element"] = element;
					result["size"] = std::stoul(literal->value());
					sizeResolved = true;
				}
				catch (...)
				{
					// Non-integer literal — will try resolved type below
				}
			}
			// If the length is a computed expression (e.g. 2**10 + 3),
			// use the resolved type annotation to get the actual size
			if (!sizeResolved)
			{
				auto const* resolvedArrayType = dynamic_cast<ArrayType const*>(arrayType->annotation().type);
				if (resolvedArrayType && !resolvedArrayType->isDynamicallySized())
				{
					u256 len = resolvedArrayType->length();
					if (len <= std::numeric_limits<unsigned long>::max())
					{
						result["kind"] = "fixed_array";
						result["element"] = element;
						result["size"] = len.convert_to<unsigned long>();
						sizeResolved = true;
					}
				}
			}
			// Final fallback: treat as dynamic array
			if (!sizeResolved)
			{
				result["kind"] = "array";
				result["element"] = element;
			}
		}
		else
		{
			// Dynamic array: uint256[]
			result["kind"] = "array";
			result["element"] = element;
		}
		return result;
	}
	if (auto const* userDefined = dynamic_cast<UserDefinedTypeName const*>(&_typeName))
	{
		auto const& path = userDefined->namePath();
		if (path.empty())
			throw UnsupportedSolCore("User-defined type without name path.");
		Declaration const* referencedDecl = userDefined->pathNode().annotation().referencedDeclaration;
		// Check if this references a contract or interface — these are address types in the EVM
		if (referencedDecl && dynamic_cast<ContractDefinition const*>(referencedDecl))
			return Json("address");
		// Check for user-defined value types (type MyUInt8 is uint8) — resolve to underlying type
		if (auto const* udtDef = dynamic_cast<UserDefinedValueTypeDefinition const*>(referencedDecl))
			return exportTypeName(*udtDef->underlyingType());
		// Check if this user-defined type is an enum
		if (referencedDecl && dynamic_cast<EnumDefinition const*>(referencedDecl))
		{
			auto const* enumDef = dynamic_cast<EnumDefinition const*>(referencedDecl);
			Json result = Json::object();
			result["kind"] = "enum";
			result["name"] = exportedEnumName(*enumDef);
			return result;
		}
		Json result = Json::object();
		result["kind"] = "named";
		result["name"] = std::string(path.back());
		return result;
	}
	// Function types (e.g., function() external) — treated as U256 at EVM level
	// (function pointers are 24-byte values: address + selector)
	if (dynamic_cast<FunctionTypeName const*>(&_typeName))
		return Json("u256");
	throw UnsupportedSolCore("Unsupported type node in SolCore exporter.");
}

/// Create a default (zero) value expression for a given type name.
/// Used for initializing named return variables.
Json defaultValueForTypeName(TypeName const& _typeName)
{
	if (auto const* elementary = dynamic_cast<ElementaryTypeName const*>(&_typeName))
	{
		switch (elementary->typeName().token())
		{
		case Token::Bool:
		{
			Json result = Json::object();
			result["kind"] = "bool";
			result["value"] = false;
			return result;
		}
		case Token::Address:
		{
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = "0";
			return result;
		}
		default:
		{
			// All integer types default to 0
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = "0";
			return result;
		}
		}
	}
	// Fallback: zero as u256
	Json result = Json::object();
	result["kind"] = "u256";
	result["value"] = "0";
	return result;
}

Json exportParam(VariableDeclaration const& _decl)
{
	Json result = Json::object();
	result["name"] = _decl.name().empty() ? ("arg" + std::to_string(_decl.id())) : _decl.name();
	result["type"] = exportTypeName(_decl.typeName());
	return result;
}

Json exportField(VariableDeclaration const& _decl)
{
	Json result = Json::object();
	result["name"] = _decl.name();
	result["type"] = exportTypeName(_decl.typeName());
	return result;
}

Json exportStorageField(
	CompilerStack const& _compilerStack,
	ContractDefinition const& _contract,
	VariableDeclaration const& _decl)
{
	Json result = exportField(_decl);
	if (auto slot = lookupStorageSlot(_compilerStack, _contract, _decl.name()))
		result["storageSlot"] = *slot;
	return result;
}

Json exportEventParam(VariableDeclaration const& _decl)
{
	Json result = Json::object();
	result["name"] = _decl.name().empty() ? ("arg" + std::to_string(_decl.id())) : _decl.name();
	result["type"] = exportTypeName(_decl.typeName());
	result["indexed"] = _decl.isIndexed();
	return result;
}

Json exportEvent(EventDefinition const& _event)
{
	Json result = Json::object();
	result["name"] = _event.name();
	result["anonymous"] = _event.isAnonymous();
	result["params"] = Json::array();
	for (auto const& parameter: _event.parameters())
		result["params"].emplace_back(exportEventParam(*parameter));
	return result;
}

std::string runtimeFieldForMagicMember(std::string const& _base, std::string const& _member)
{
	if (_base == "msg")
	{
		if (_member == "sender")
			return "msgSender";
		if (_member == "value")
			return "msgValue";
		if (_member == "data")
			return "calldata";
	}
	if (_base == "block")
	{
		if (_member == "timestamp")
			return "blockTimestamp";
		if (_member == "number")
			return "blockNumber";
		if (_member == "chainid")
			return "chainId";
	}
	throw UnsupportedSolCore("Unsupported magic member access: " + _base + "." + _member);
}

Json exportExpr(Expression const& _expr);

/// [SolCore audit finding #10] Whether `_expr` denotes this contract's own
/// address, i.e. is (possibly wrapped in one or more no-op `address(...)`/
/// `payable(...)` type conversions around) the magic identifier `this`. Used
/// to distinguish `address(this).balance` (correctly modeled as the shared
/// world's `contractBalance`, which `syncLegacyContractBalance` keeps aligned
/// with the currently-executing address) from `X.balance` for any other
/// address `X`, which must NOT take that shortcut (see the `.balance`
/// handling below).
bool isThisAddressExpr(Expression const& _expr)
{
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
		return dynamic_cast<MagicVariableDeclaration const*>(identifier->annotation().referencedDeclaration)
			&& identifier->name() == "this";
	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
		if (call->annotation().kind.set() && *call->annotation().kind == FunctionCallKind::TypeConversion
			&& call->arguments().size() == 1 && call->arguments().front())
			return isThisAddressExpr(*call->arguments().front());
	return false;
}

Json u256Literal(std::string const& _value)
{
	Json result = Json::object();
	result["kind"] = "u256";
	result["value"] = _value;
	return result;
}

Json localExpr(std::string const& _name)
{
	Json result = Json::object();
	result["kind"] = "local";
	result["name"] = _name;
	return result;
}

std::optional<Json> exportExternalContractCall(
	FunctionCall const& _call,
	MemberAccess const& _memberAccess,
	bool _asStatement
)
{
	Type const* baseType = _memberAccess.expression().annotation().type;
	if (!baseType || baseType->category() != Type::Category::Contract)
		return std::nullopt;

	Json callExpr = Json::object();
	callExpr["target"] = exportExpr(_memberAccess.expression());
	callExpr["method"] = _memberAccess.memberName();
	callExpr["args"] = Json::array();
	for (auto const& arg: _call.arguments())
		callExpr["args"].emplace_back(exportExpr(*arg));

	bool statefulCall = true;
	if (activeCompilerStack)
	{
		if (auto knownTarget = exportKnownExternalTarget(*activeCompilerStack, _memberAccess))
		{
			callExpr["knownTarget"] = *knownTarget;
			if (
				knownTarget->contains("mutability") &&
				(*knownTarget)["mutability"].is_string() &&
				(*knownTarget)["mutability"].get<std::string>() == "view"
			)
				statefulCall = false;
		}
	}

	if (_asStatement && statefulCall)
	{
		callExpr["kind"] = "external_call_stmt";
		return callExpr;
	}

	callExpr["kind"] = statefulCall ? "external_call_effect" : "external_call";
	if (!_asStatement)
		return callExpr;

	Json result = Json::object();
	result["kind"] = "expr";
	result["value"] = callExpr;
	return result;
}

std::string sanitizeExportNameComponent(std::string const& _value)
{
	std::string sanitized;
	sanitized.reserve(_value.size());
	bool lastWasUnderscore = false;
	for (char c: _value)
	{
		unsigned char uc = static_cast<unsigned char>(c);
		if (std::isalnum(uc))
		{
			sanitized.push_back(c);
			lastWasUnderscore = false;
		}
		else if (!lastWasUnderscore)
		{
			sanitized.push_back('_');
			lastWasUnderscore = true;
		}
	}
	while (!sanitized.empty() && sanitized.back() == '_')
		sanitized.pop_back();
	if (sanitized.empty())
		return "arg";
	return sanitized;
}

std::string overloadSignatureSuffix(FunctionDefinition const& _function)
{
	if (_function.parameters().empty())
		return "unit";

	std::string suffix;
	for (auto const& parameter: _function.parameters())
	{
		if (!suffix.empty())
			suffix += "_";
		std::string typeName =
			parameter->annotation().type ?
				parameter->annotation().type->toString() :
				("arg" + std::to_string(parameter->id()));
		suffix += sanitizeExportNameComponent(typeName);
	}
	return suffix;
}

std::string exportedFunctionName(FunctionDefinition const& _function)
{
	auto it = exportedFunctionNames.find(&_function);
	if (it != exportedFunctionNames.end())
		return it->second;
	std::string rawName = _function.name().empty() ? "_unnamed" : _function.name();
	ContractDefinition const* contract = _function.annotation().contract;
	if (!contract)
		return rawName;
	std::set<std::string> matchingSignatures;
	for (FunctionDefinition const* candidate: contract->definedFunctions())
	{
		if (
			candidate &&
			candidate->isOrdinary() &&
			candidate->isImplemented() &&
			(candidate->name().empty() ? "_unnamed" : candidate->name()) == rawName
		)
			matchingSignatures.insert(overloadSignatureSuffix(*candidate));
	}
	if (matchingSignatures.size() > 1)
		return rawName + "_" + overloadSignatureSuffix(_function);
	return rawName;
}

std::string contractScopedSuperAlias(FunctionDefinition const& _function)
{
	auto const* owner = dynamic_cast<ContractDefinition const*>(_function.scope());
	std::string ownerName =
		owner && !owner->name().empty() ? owner->name() : "Base";
	return (_function.name().empty() ? "_unnamed" : _function.name()) +
		"__super__" + sanitizeExportNameComponent(ownerName);
}

/// A statically-bound base-contract call resolved against the most-derived
/// (exporting) contract's C3 linearization. `needsAlias == false` means the
/// resolved target IS the most-derived implementation of its virtual slot, so
/// the plain exported slot name already denotes exactly this body and no
/// flattened `f__super__Owner` sibling is required.
struct StaticBaseCallTarget
{
	FunctionDefinition const* target;
	bool needsAlias;
};

/// Classifies a member-access function call as a statically-bound internal
/// call through a contract qualifier — `super.f(...)` or `Base.f(...)` — and
/// resolves the EXACT implementation it must dispatch to, following solc's
/// own dispatch rules (mirrors ASTNode::resolveFunctionCall):
///   - `super.f(...)`: virtual lookup in _mostDerived's linearization
///     starting strictly after the contract that lexically contains the call
///     (NOT the raw referencedDeclaration annotation, which was resolved
///     against the DEFINING contract's own linearization and is wrong
///     whenever the most-derived contract interposes another override — the
///     inherited-override case, e.g. StRSRP1Votes.beginEra emitted inside
///     StRSRP1VotesV2's artifact).
///   - `Base.f(...)` (VirtualLookup::Static): bound exactly to the referenced
///     declaration. Previously this shape fell through to the generic
///     "library/type-qualified" lowering, whose exportedFunctionName collapses
///     every same-name/same-signature function in the hierarchy onto the
///     virtual-slot name — so `GovernorTimelockControl.state(id)` inside
///     Governance.state lowered to a bare self-named `internal_call state`,
///     silently dropping GovernorTimelockControl.state's refinements.
///
/// Returns std::nullopt when the call is NOT this shape (library-qualified
/// calls, using-for receivers, external contract calls, magic builtins,
/// struct constructors, type conversions — all handled by pre-existing
/// paths). Once the shape matches, this function either resolves the target
/// or throws UnsupportedSolCore (fail-closed): it never lets the call fall
/// through to a lowering that would emit a silent self-forward.
std::optional<StaticBaseCallTarget> resolveStaticBaseCallTarget(
	FunctionCall const& _call,
	MemberAccess const& _memberAccess,
	ContractDefinition const* _mostDerived)
{
	auto const* typeType =
		dynamic_cast<TypeType const*>(_memberAccess.expression().annotation().type);
	if (!typeType)
		return std::nullopt;
	auto const* qualifierType = dynamic_cast<ContractType const*>(typeType->actualType());
	if (!qualifierType)
		return std::nullopt;
	if (qualifierType->contractDefinition().isLibrary())
		return std::nullopt;
	// Struct constructors (`Base.S(...)`) and type conversions are not
	// function dispatch; leave them to their dedicated handlers.
	if (_call.annotation().kind.set() && *_call.annotation().kind != FunctionCallKind::FunctionCall)
		return std::nullopt;

	// From here on this IS a statically-bound super/base-qualified internal
	// call: resolve it exactly or fail closed.
	auto const* funcDef =
		dynamic_cast<FunctionDefinition const*>(_memberAccess.annotation().referencedDeclaration);
	if (!funcDef)
		throw UnsupportedSolCore(
			"Base-qualified call '" + _memberAccess.memberName() +
			"' does not reference a function definition.");
	if (!_mostDerived)
		throw UnsupportedSolCore(
			"Base-qualified call to '" + funcDef->name() +
			"' outside of a contract export context.");

	FunctionDefinition const* target = nullptr;
	if (qualifierType->isSuper())
	{
		// The contract of a super type is the contract lexically containing
		// the call (annotations are assigned once, when that contract was
		// type-checked). Solidity semantics: search _mostDerived's MRO
		// strictly after it.
		ContractDefinition const& definingContract = qualifierType->contractDefinition();
		auto const& hierarchy = _mostDerived->annotation().linearizedBaseContracts;
		if (std::find(hierarchy.begin(), hierarchy.end(), &definingContract) == hierarchy.end())
			throw UnsupportedSolCore(
				"super call in '" + definingContract.name() + "' but '" +
				definingContract.name() + "' is not a base of '" + _mostDerived->name() + "'.");
		ContractDefinition const* searchStart = definingContract.superContract(*_mostDerived);
		if (!searchStart)
			throw UnsupportedSolCore(
				"super call in '" + definingContract.name() +
				"' with no remaining base contracts in '" + _mostDerived->name() + "'.");
		target = &funcDef->resolveVirtual(*_mostDerived, searchStart);
	}
	else
		// Explicit `Base.f(...)` is statically bound to the referenced
		// declaration (VirtualLookup::Static) — no virtual lookup.
		target = funcDef;

	if (!target->isOrdinary() || !target->isImplemented())
		throw UnsupportedSolCore(
			"Statically-bound base call to '" + target->name() +
			"' has no implemented ordinary target.");

	// If the statically-bound target is ALSO the most-derived implementation
	// of its virtual slot, the plain exported slot name denotes exactly this
	// body: emit a plain call instead of aliasing (aliasing would rename the
	// slot's only implementation and orphan every plain-name caller).
	bool needsAlias = &target->resolveVirtual(*_mostDerived) != target;
	return StaticBaseCallTarget{target, needsAlias};
}

/// The exported callee name for a resolved statically-bound base call.
/// For aliased targets, verifies the flattening pass actually registered the
/// `f__super__Owner` sibling (exportContract registers every function
/// collected by collectSuperReferencedFunctions, which uses the same resolver
/// — a mismatch means exporter drift and must fail closed, not emit a bare
/// slot name that self-forwards).
std::string flattenedStaticBaseCallName(StaticBaseCallTarget const& _resolved)
{
	if (!_resolved.needsAlias)
		return exportedFunctionName(*_resolved.target);
	std::string alias = contractScopedSuperAlias(*_resolved.target);
	auto it = exportedFunctionNames.find(_resolved.target);
	if (it == exportedFunctionNames.end() || it->second != alias)
		throw UnsupportedSolCore(
			"Statically-bound base call target '" + alias +
			"' was not registered for flattened export.");
	return alias;
}

/// The exported callee name for a PLAIN (unqualified-identifier) internal
/// call — Solidity VIRTUAL dispatch (mirrors ASTNode::resolveFunctionCall's
/// Identifier arm): the call must reach the most-derived override in the
/// exporting contract's linearization, i.e. the exported virtual-slot name.
/// Using exportedFunctionName(referencedDeclaration) directly is wrong here:
/// the lexically referenced declaration (e.g. Governor.state inside
/// Governor.execute's body) may be registered under a flattened
/// `f__super__Owner` alias because some OTHER call site super/base-qualifies
/// it — a plain call would then silently bind to the base implementation and
/// skip every more-derived override (e.g. execute() skipping
/// GovernorTimelockControl.state's timelock refinement).
std::string virtualCallTargetName(FunctionDefinition const& _funcDef)
{
	FunctionDefinition const* target = &_funcDef;
	if (
		activeExportContract &&
		_funcDef.isOrdinary() &&
		!_funcDef.name().empty() &&
		_funcDef.virtualSemantics()
	)
		// The resolved slot winner is by construction never alias-registered
		// (resolveStaticBaseCallTarget only flattens targets that are NOT
		// their slot's most-derived implementation), so this lookup yields
		// the plain exported slot name.
		target = &_funcDef.resolveVirtual(*activeExportContract);
	return exportedFunctionName(*target);
}

std::set<FunctionDefinition const*> collectSuperReferencedFunctions(
	ContractDefinition const& _contract)
{
	struct SuperReferenceCollector: ASTConstVisitor
	{
		ContractDefinition const& mostDerived;
		std::set<FunctionDefinition const*> functions;

		explicit SuperReferenceCollector(ContractDefinition const& _mostDerived):
			mostDerived(_mostDerived)
		{}

		bool visit(FunctionCall const& _call) override
		{
			auto const* memberAccess =
				dynamic_cast<MemberAccess const*>(&_call.expression());
			if (!memberAccess)
				return true;
			try
			{
				// Same resolver the call-site lowering uses (exportExpr /
				// exportStmt), so the alias a call site emits and the sibling
				// this collection causes to be flattened can never diverge.
				// Covers both `super.f(...)` AND explicitly-qualified
				// `Base.f(...)` calls.
				if (auto resolved = resolveStaticBaseCallTarget(_call, *memberAccess, &mostDerived))
					if (resolved->needsAlias)
						functions.insert(resolved->target);
			}
			catch (...)
			{
				// Unresolvable statically-bound call: the matching throw in
				// the call-site lowering will independently fail the
				// containing body CLOSED (kind:"unsupported_body") when it is
				// exported, so there is no sibling to register here.
			}
			return true;
		}
	};

	std::set<FunctionDefinition const*> referenced;
	std::set<FunctionDefinition const*> visited;
	std::vector<FunctionDefinition const*> worklist;
	// Seed over the full linearized base hierarchy (MRO), not just functions
	// defined directly in this contract: a super.X() call can live inside the
	// body of an INHERITED function, which definedFunctions() does not return.
	// The visited set below de-duplicates across bases.
	for (ContractDefinition const* base: _contract.annotation().linearizedBaseContracts)
	{
		for (FunctionDefinition const* function: base->definedFunctions())
			if (function && function->isOrdinary() && function->isImplemented())
				worklist.push_back(function);
		// Constructors are walked as SEEDS only (they are not ordinary, are
		// never flattened as siblings themselves, and the targets the
		// resolver returns are guaranteed ordinary) — a constructor body may
		// contain `Base.f(...)`/`super.f(...)` calls whose flattened siblings
		// the artifact must include.
		if (FunctionDefinition const* ctor = base->constructor())
			if (ctor->isImplemented())
				worklist.push_back(ctor);
	}

	while (!worklist.empty())
	{
		FunctionDefinition const* function = worklist.back();
		worklist.pop_back();
		if (!function || visited.count(function))
			continue;
		visited.insert(function);

		SuperReferenceCollector collector{_contract};
		function->body().accept(collector);
		for (FunctionDefinition const* referencedFunction: collector.functions)
		{
			if (!referencedFunction || !referencedFunction->isOrdinary() || !referencedFunction->isImplemented())
				continue;
			if (referenced.insert(referencedFunction).second)
				worklist.push_back(referencedFunction);
		}
	}

	return referenced;
}

void assignExportedFunctionNames(std::vector<FunctionDefinition const*> const& _functions)
{
	exportedFunctionNames.clear();

	std::map<std::string, std::vector<FunctionDefinition const*>> byName;
	for (FunctionDefinition const* function: _functions)
	{
		if (!function)
			continue;
		byName[function->name().empty() ? "_unnamed" : function->name()].push_back(function);
	}

	for (auto const& [rawName, group]: byName)
	{
		std::map<std::string, std::vector<FunctionDefinition const*>> bySignature;
		for (FunctionDefinition const* function: group)
			bySignature[overloadSignatureSuffix(*function)].push_back(function);

		if (bySignature.size() <= 1)
		{
			for (FunctionDefinition const* function: group)
				exportedFunctionNames[function] = rawName;
			continue;
		}

		std::map<std::string, int> usedCandidates;
		for (auto const& [signatureSuffix, functions]: bySignature)
		{
			std::string candidate = rawName + "_" + signatureSuffix;
			int& count = usedCandidates[candidate];
			if (count > 0)
				candidate += "_" + std::to_string(count);
			++count;
			for (FunctionDefinition const* function: functions)
				exportedFunctionNames[function] = candidate;
		}
	}
}

std::pair<std::vector<std::string>, Json> exportStorageMapLValue(Expression const& _expr)
{
	auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr);
	if (!indexAccess || !indexAccess->indexExpression())
		throw UnsupportedSolCore("Expected single-level mapping index access.");

	Expression const& base = indexAccess->baseExpression();

	// Check for namespaced storage: $.mappingField[key]
	// base is MemberAccess on a namespaced alias
	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&base))
	{
		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			std::string baseName = baseIdent->name();
			auto it = namespacedStorageAliases.find(baseName);
			if (it != namespacedStorageAliases.end())
			{
				std::string fieldName = it->second + memberAccess->memberName();
				return {{fieldName}, exportExpr(*indexAccess->indexExpression())};
			}
		}
		// Also check for _getTokenStorage().mappingField[key] pattern
		if (auto const* call = dynamic_cast<FunctionCall const*>(&memberAccess->expression()))
		{
			if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
			{
				if (auto const* prefix = namespacedGetterPrefix(*callee))
				{
					std::string fieldName = *prefix + memberAccess->memberName();
					return {{fieldName}, exportExpr(*indexAccess->indexExpression())};
				}
			}
		}
	}

	if (auto const* identifier = dynamic_cast<Identifier const*>(&base))
	{
		auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (decl && decl->isStateVariable())
			return {{decl->name()}, exportExpr(*indexAccess->indexExpression())};
	}

	throw UnsupportedSolCore("Only direct state-mapping index access is supported.");
}

// A bounded, well-known set of OpenZeppelin `SafeCast`/`SafeCastUpgradeable`
// narrowing-downcast helpers (`toUint8`..`toUint248`, `toInt8`..`toInt248`,
// including the commonly-used `toUint128`/`toUint240`/etc.) whose return
// value can be soundly attributed to the call site's own argument TRUNCATED
// to the target width, using the exact same `"truncate"` JSON node the
// inline narrowing-downcast case (`uint128(x)`, below) already emits
// (SOLCORE_MATH_BUG_CLASSES_PLAN.md §3.5/§6.3, Task 11) — instead of
// exporting the call as an opaque `internal_call` to a function this
// exporter never walks into: `SafeCast` is a library, never a base contract,
// so its body is never part of `contract.definedFunctions()`/
// `linearizedBaseContracts` (the main per-contract export loop, far below,
// only walks those two sets), and prior to this the OCaml frontend's only
// handling of an unresolved `toUintN`/`toIntN`-named internal_call was a
// name-prefix match used purely for TYPE inference (`Analysis.ml`'s
// `infer_builtin_internal_return_ty`) — the call itself still lowered to a
// fully opaque, unconstrained-return-value runtime helper.
//
// `SafeCast.toUintN`/`toIntN` also has a `require`/custom-error revert when
// the value is out of range, which this (like the inline downcast case)
// does not model — the same pre-existing, explicitly out-of-scope gap noted
// in `Arith/Truncation.lean`'s "Explicit downcasts" comment. On every path
// that does NOT revert, though, its return value is bit-for-bit identical
// to `uint128(x)`'s: a `mod 2^n` truncation of the input. Attributing
// exactly that (and only that) is sound and a strict precision improvement
// over the fully-opaque model it replaces.
bool isKnownOzSafeCastNarrowingDowncast(FunctionDefinition const& _funcDef, int& _targetBits)
{
	auto const* library = dynamic_cast<ContractDefinition const*>(_funcDef.scope());
	if (!library || !library->isLibrary())
		return false;
	std::string const& libName = library->name();
	if (libName != "SafeCast" && libName != "SafeCastUpgradeable")
		return false;
	std::string const& fnName = _funcDef.name();
	if (fnName.rfind("toUint", 0) != 0 && fnName.rfind("toInt", 0) != 0)
		return false;
	// Defensive shape check: a genuine narrowing downcast takes exactly one
	// integer-typed argument and returns a STRICTLY NARROWER integer type.
	// This is also what excludes SafeCast's OTHER same-width
	// sign-reinterpretation helpers that happen to share the
	// `to(Uint|Int)NNN` naming shape — `toUint256(int256)` and
	// `toInt256(uint256)` — which are not truncations at all (both operands
	// are 256 bits; the check those two perform is a sign/range check, not
	// a bit-width narrowing), so falling through to the generic
	// (fail-closed, opaque) path for them is correct, not a gap this
	// function needs to close.
	if (_funcDef.parameters().size() != 1 || _funcDef.returnParameters().size() != 1)
		return false;
	Type const* sourceType = _funcDef.parameters().front()->type();
	Type const* targetType = _funcDef.returnParameters().front()->type();
	if (
		!sourceType || !targetType ||
		sourceType->category() != Type::Category::Integer ||
		targetType->category() != Type::Category::Integer
	)
		return false;
	auto const* sourceInt = dynamic_cast<IntegerType const*>(sourceType);
	auto const* targetInt = dynamic_cast<IntegerType const*>(targetType);
	if (!sourceInt || !targetInt || targetInt->numBits() >= sourceInt->numBits())
		return false;
	_targetBits = static_cast<int>(targetInt->numBits());
	return true;
}

Json exportExpr(Expression const& _expr)
{
	if (auto const* literal = dynamic_cast<Literal const*>(&_expr))
	{
		if (literal->token() == Token::TrueLiteral)
		{
			Json result = Json::object();
			result["kind"] = "bool";
			result["value"] = true;
			return result;
		}
		if (literal->token() == Token::FalseLiteral)
		{
			Json result = Json::object();
			result["kind"] = "bool";
			result["value"] = false;
			return result;
		}
		if (literal->annotation().type && literal->annotation().type->category() == Type::Category::RationalNumber)
		{
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = literal->value();
			return result;
		}
		// Address literals: 0x... with 40 hex digits
		if (literal->annotation().type && literal->annotation().type->category() == Type::Category::Address)
		{
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = literal->value();
			return result;
		}
		// Hex string literals (e.g. hex"0f") are byte sequences.
		// In our model, all bytesN types are U256, so convert to a big-endian integer.
		if (literal->token() == Token::HexStringLiteral)
		{
			std::string const& raw = literal->value();
			u256 val = 0;
			for (size_t i = 0; i < raw.size(); ++i)
				val = val * 256 + static_cast<unsigned char>(raw[i]);
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = val.str();
			return result;
		}
		if (literal->token() == Token::StringLiteral || literal->token() == Token::UnicodeStringLiteral)
		{
			Json result = Json::object();
			result["kind"] = "string";
			// Hex-encode string values that contain non-ASCII/non-printable bytes
			// to avoid breaking JSON serialization (nlohmann/json's dump() with
			// ensure_ascii=true throws on invalid UTF-8 sequences).
			std::string const& raw = literal->value();
			bool needsHexEncoding = false;
			for (char c : raw)
				if (static_cast<unsigned char>(c) > 126 || static_cast<unsigned char>(c) < 32)
				{
					needsHexEncoding = true;
					break;
				}
			if (needsHexEncoding)
			{
				static char const digits[] = "0123456789abcdef";
				std::string hex;
				hex.reserve(raw.size() * 2);
				for (size_t i = 0; i < raw.size(); ++i)
				{
					auto c = static_cast<unsigned char>(raw[i]);
					hex += digits[c >> 4];
					hex += digits[c & 0xf];
				}
				result["value"] = hex;
				result["encoding"] = "hex";
			}
			else
				result["value"] = raw;
			return result;
		}
		throw UnsupportedSolCore("Only boolean, u256, and string literals are currently supported.");
	}

	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
	{
		// Check if the identifier directly references an enum value (rare but possible)
		if (auto const* enumValue = dynamic_cast<EnumValue const*>(identifier->annotation().referencedDeclaration))
		{
			auto const* enumDef = dynamic_cast<EnumDefinition const*>(enumValue->scope());
			if (enumDef)
			{
				Json result = Json::object();
				result["kind"] = "enum_variant";
				result["enum"] = exportedEnumName(*enumDef);
				result["variant"] = enumValue->name();
				return result;
			}
		}
		if (auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration))
		{
			// For constant/immutable variables, try to inline the value.
			// Guard against infinite recursion from self-referential initializers
			// like `uint immutable x = x + 1` by tracking inlining depth.
			static thread_local int inlineDepth = 0;
			if ((decl->isConstant() || decl->immutable()) && decl->value() && inlineDepth < 3)
			{
				try
				{
					++inlineDepth;
					auto result = exportExpr(*decl->value());
					--inlineDepth;
					return result;
				}
				catch (...)
				{
					--inlineDepth;
					// For file-level / non-state constants that couldn't be inlined
					// directly (e.g. `IERC20 constant CRV = IERC20(0xD533...)`),
					// try to extract the literal address from the type-conversion
					// wrapper: the value is a FunctionCall whose argument is a literal.
					if (decl->isConstant() && !decl->isStateVariable() && decl->value())
					{
						// Pattern: TypeConversion(Literal) e.g. IERC20(0xD533...)
						if (auto const* call = dynamic_cast<FunctionCall const*>(decl->value().get()))
						{
							if (*call->annotation().kind == FunctionCallKind::TypeConversion
								&& call->arguments().size() == 1)
							{
								try
								{
									++inlineDepth;
									auto inner = exportExpr(*call->arguments().front());
									--inlineDepth;
									return inner;
								}
								catch (...)
								{
									--inlineDepth;
								}
							}
						}
						// Pattern: Literal address/number
						if (auto const* literal = dynamic_cast<Literal const*>(decl->value().get()))
						{
							(void)literal;
							// Already tried above via exportExpr, but try the annotation
							Type const* annType = decl->value()->annotation().type;
							if (annType)
							{
								if (annType->category() == Type::Category::Address
									|| annType->category() == Type::Category::RationalNumber
									|| annType->category() == Type::Category::Integer)
								{
									Json result = Json::object();
									result["kind"] = "u256";
									result["value"] = annType->toString(true);
									return result;
								}
							}
						}
					}
					// Fall through to storage_get / local below
				}
			}
			Json result = Json::object();
			if (decl->isStateVariable())
			{
				result["kind"] = "storage_get";
				result["field"] = decl->name();
				return result;
			}
			result["kind"] = "local";
			result["name"] = decl->name().empty() ? identifier->name() : decl->name();
			return result;
		}
		// Handle magic variables like "this" — they are not locals
		if (auto const* magicDecl = dynamic_cast<MagicVariableDeclaration const*>(identifier->annotation().referencedDeclaration))
		{
			(void)magicDecl;
			if (identifier->name() == "this")
			{
				// "this" refers to the current contract's address
				Json result = Json::object();
				result["kind"] = "state_get";
				result["path"] = jsonStringArray({"env", "thisAddress"});
				return result;
			}
			throw UnsupportedSolCore("Unsupported magic variable: " + identifier->name());
		}
		// [SolCore audit finding #18] Each of the four cases below used to
		// emit a silent `{u256,0}` placeholder for an identifier this
		// exporter has no real value-context lowering for -- a real
		// occurrence would previously read as the literal number 0 to every
		// downstream consumer with no trace that anything was substituted
		// (e.g. `require(x != address(SomeLibrary))` would tautologically
		// compare against 0 instead of failing to export). Fail closed
		// instead: the (rare) real occurrence becomes a diagnosable
		// "unsupported_body" for its containing function rather than a
		// silently-wrong value baked into a proof.

		// Identifiers that reference contracts/interfaces/libraries used
		// directly as a value (e.g. a bare library/interface name outside
		// of `address(...)`/a qualified member access, both handled above).
		if (dynamic_cast<ContractDefinition const*>(identifier->annotation().referencedDeclaration))
			throw UnsupportedSolCore(
				"A bare contract/interface/library identifier used as a "
				"value has no real address representation in the SolCore "
				"exporter.");
		// Identifiers that reference function definitions used as values
		// (internal function pointers) -- not modeled at all.
		if (dynamic_cast<FunctionDefinition const*>(identifier->annotation().referencedDeclaration))
			throw UnsupportedSolCore(
				"An internal function used as a value (a function pointer) "
				"is not modeled by the SolCore exporter.");
		// Identifiers that reference user-defined value type definitions
		// (e.g. the bare `MyAddress` in `MyAddress.wrap(...)`/`.unwrap(...)`,
		// used as a value rather than as the base of a wrap/unwrap call).
		if (dynamic_cast<UserDefinedValueTypeDefinition const*>(identifier->annotation().referencedDeclaration))
			throw UnsupportedSolCore(
				"A user-defined value type name used as a value has no "
				"real representation in the SolCore exporter.");
		// Any other non-variable declaration used in expression context
		// (e.g., struct name, error name, event name): these are
		// type/declaration-level references, not runtime values, and the
		// VariableDeclaration case was already handled above.
		if (identifier->annotation().referencedDeclaration)
			throw UnsupportedSolCore(
				"'" + identifier->name() +
				"' is a declaration-level reference (not a variable) used "
				"in value context, which the SolCore exporter cannot "
				"represent as a real value.");
		// No declaration resolved at all for this identifier. Every genuine
		// local-variable reference resolves to a VariableDeclaration above,
		// so reaching here means name resolution didn't attach a
		// declaration -- guessing "local" would silently synthesize a
		// reference to a variable that may never have been declared in the
		// translated scope. Fail closed rather than guess.
		throw UnsupportedSolCore(
			"Identifier '" + identifier->name() +
			"' has no resolved declaration; the SolCore exporter cannot "
			"determine what value it refers to.");
	}

	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
	{
		// --- Namespaced storage: $.field → storage_get with prefixed field ---
		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			auto it = namespacedStorageAliases.find(baseIdent->name());
			if (it != namespacedStorageAliases.end())
			{
				std::string prefixedField = it->second + memberAccess->memberName();
				// Check if this field is a mapping — if so, don't emit storage_get here,
				// let IndexAccess handle it. But if used standalone (e.g. as base for length),
				// emit storage_get.
				Json result = Json::object();
				result["kind"] = "storage_get";
				result["field"] = prefixedField;
				return result;
			}
		}
		// --- Namespaced storage: _getTokenStorage().field → storage_get ---
		if (auto const* call = dynamic_cast<FunctionCall const*>(&memberAccess->expression()))
		{
			if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
			{
				if (auto const* prefix = namespacedGetterPrefix(*callee))
				{
					std::string prefixedField = *prefix + memberAccess->memberName();
					Json result = Json::object();
					result["kind"] = "storage_get";
					result["field"] = prefixedField;
					return result;
				}
			}
		}
		// --- End namespaced storage member access ---

		// Check if this member access resolves to an enum value.
		// This handles ALL forms: Direction.Right, I.Direction.Right, L.Direction.Right
		// because the Solidity type checker resolves the member access to the actual EnumValue declaration.
		if (memberAccess->annotation().referencedDeclaration)
		{
			auto const* enumValue = dynamic_cast<EnumValue const*>(memberAccess->annotation().referencedDeclaration);
			if (enumValue)
			{
				auto const* enumDef = dynamic_cast<EnumDefinition const*>(enumValue->scope());
				if (enumDef)
				{
					Json result = Json::object();
					result["kind"] = "enum_variant";
					result["enum"] = exportedEnumName(*enumDef);
					result["variant"] = enumValue->name();
					return result;
				}
			}
		}

		if (auto const* baseIdentifier = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			auto const* decl = dynamic_cast<MagicVariableDeclaration const*>(baseIdentifier->annotation().referencedDeclaration);
			if (decl)
			{
				Json result = Json::object();
				result["kind"] = "state_get";
				result["path"] = jsonStringArray({"env", runtimeFieldForMagicMember(baseIdentifier->name(), memberAccess->memberName())});
				return result;
			}
		}

		// ContractName.stateVar access — accessing state variable through contract type qualifier
		if (memberAccess->annotation().referencedDeclaration)
		{
			if (auto const* varDecl = dynamic_cast<VariableDeclaration const*>(memberAccess->annotation().referencedDeclaration))
			{
				if (varDecl->isStateVariable())
				{
					// Constant state variables — inline the value
					if (varDecl->isConstant() && varDecl->value())
						return exportExpr(*varDecl->value());
					// Immutable state variables with compile-time value — try to inline
					if (varDecl->immutable() && varDecl->value())
					{
						try
						{
							return exportExpr(*varDecl->value());
						}
						catch (...)
						{
							// Fall through to storage_get below
						}
					}
					// Regular state variable — treat as storage_get
					Json result = Json::object();
					result["kind"] = "storage_get";
					result["field"] = varDecl->name();
					return result;
				}
			}
		}

		// address.balance access
		if (memberAccess->memberName() == "balance")
		{
			Type const* baseType = memberAccess->expression().annotation().type;
			if (baseType && baseType->category() == Type::Category::Address)
			{
				// [SolCore audit finding #10] This used to unconditionally
				// return the shared world's `contractBalance` regardless of
				// which address's `.balance` was actually accessed, so
				// `address(attacker).balance`, `token.balance`, and
				// `address(this).balance` all silently collapsed to THIS
				// contract's own balance. `contractBalance` genuinely is
				// this contract's balance (kept in sync by
				// `syncLegacyContractBalance`), so the shortcut is correct
				// ONLY for `address(this).balance` (or an equivalent chain
				// of no-op `address(...)`/`payable(...)` conversions around
				// `this`). For any other address expression there is no
				// per-address balance lookup this exporter can wire up yet
				// (the Lean runtime's `WorldState.balances`/`getBalance`
				// support querying an arbitrary address, but nothing on the
				// OCaml-frontend side consumes such a node), so fail closed
				// instead of silently reading the wrong account's balance.
				if (isThisAddressExpr(memberAccess->expression()))
				{
					Json result = Json::object();
					result["kind"] = "state_get";
					result["path"] = jsonStringArray({"world", "contractBalance"});
					return result;
				}
				throw UnsupportedSolCore(
					"'.balance' on an address other than address(this) is not "
					"modeled by the SolCore exporter yet -- it used to silently "
					"return this contract's own balance instead of the queried "
					"address's balance.");
			}
		}

		// Array .length access
		if (memberAccess->memberName() == "length")
		{
			if (auto const* codeAccess = dynamic_cast<MemberAccess const*>(&memberAccess->expression()))
			{
				Type const* codeBaseType = codeAccess->expression().annotation().type;
				if (codeAccess->memberName() == "code" && codeBaseType && codeBaseType->category() == Type::Category::Address)
				{
					Json result = Json::object();
					result["kind"] = "extcodesize";
					result["address"] = exportExpr(codeAccess->expression());
					return result;
				}
			}
			Type const* baseType = memberAccess->expression().annotation().type;
			if (baseType && baseType->category() == Type::Category::Array)
			{
				Json result = Json::object();
				result["kind"] = "array_length";
				result["base"] = exportExpr(memberAccess->expression());
				return result;
			}
			// Fixed bytes .length (e.g. bytes8(x).length) — return the byte count
			if (baseType && baseType->category() == Type::Category::FixedBytes)
			{
				auto const* fbType = dynamic_cast<FixedBytesType const*>(baseType);
				if (fbType)
				{
					Json result = Json::object();
					result["kind"] = "u256";
					result["value"] = std::to_string(fbType->numBytes());
					return result;
				}
			}
		}

		// type(X).max / type(X).min / type(X).interfaceId
		{
			Type const* baseType = memberAccess->expression().annotation().type;
			if (baseType && baseType->category() == Type::Category::Magic)
			{
				auto const* magicType = dynamic_cast<MagicType const*>(baseType);
				if (magicType && magicType->kind() == MagicType::Kind::MetaType)
				{
					Type const* typeArg = magicType->typeArgument();
					if (memberAccess->memberName() == "max" || memberAccess->memberName() == "min")
					{
						if (typeArg && typeArg->category() == Type::Category::Integer)
						{
							auto const* intType = dynamic_cast<IntegerType const*>(typeArg);
							if (intType)
							{
								Json result = Json::object();
								if (intType->isSigned())
								{
									result["kind"] = "i256";
									if (memberAccess->memberName() == "min")
										result["value"] = intType->minValue().str();
									else
										result["value"] = intType->maxValue().str();
								}
								else
								{
									result["kind"] = "u256";
									if (memberAccess->memberName() == "min")
										result["value"] = "0";
									else
										result["value"] = intType->max().str();
								}
								return result;
							}
						}
						// [SolCore audit finding #18] type(E).max/min for an enum type
						// (valid Solidity since 0.8.8) used to fall through to the
						// generic "unknown type argument" 0-placeholder below, so
						// e.g. `require(x <= type(RoundingMode).max)` silently became
						// `require(x <= 0)`. Enums are always densely numbered
						// starting at 0, so both bounds have a real, exact value:
						// `min` is always 0, and `max` is the member count minus one
						// (`EnumType::minValue()`/`maxValue()` compute exactly this).
						if (typeArg && typeArg->category() == Type::Category::Enum)
						{
							auto const* enumType = dynamic_cast<EnumType const*>(typeArg);
							if (enumType)
							{
								Json result = Json::object();
								result["kind"] = "u256";
								result["value"] = std::to_string(
									memberAccess->memberName() == "min" ?
										enumType->minValue() : enumType->maxValue());
								return result;
							}
						}
						// Genuinely unknown/unhandled type(X).max/min type argument --
						// fail closed instead of silently substituting a `0` that a
						// caller (e.g. a `require(x <= type(X).max)` bound check) would
						// then treat as a real, meaningful value.
						throw UnsupportedSolCore(
							"type(X)." + memberAccess->memberName() +
							" is not supported by the SolCore exporter for this type "
							"argument.");
					}
					if (memberAccess->memberName() == "interfaceId")
					{
						// type(SomeInterface).interfaceId — bytes4 selector XOR
						// (REV-271/bug_013/ERC165 fix: replaces a hardcoded "0" placeholder.)
						Json result = Json::object();
						result["kind"] = "u256";
						if (typeArg && typeArg->category() == Type::Category::Contract)
						{
							auto const* contractType = dynamic_cast<ContractType const*>(typeArg);
							if (contractType)
							{
								// Match codegen: shifted into top 32 bits of bytes4 word; here
								// we want the 4-byte numeric value, formatted as decimal so the
								// downstream u256 parser accepts it.
								uint32_t selector = contractType->contractDefinition().interfaceId();
								result["value"] = std::to_string(static_cast<uint64_t>(selector));
								return result;
							}
						}
						result["value"] = "0";
						return result;
					}
				}
			}
		}

		// Bytes .length on dynamic bytes
		if (memberAccess->memberName() == "length")
		{
			Type const* baseType = memberAccess->expression().annotation().type;
			if (baseType && (baseType->category() == Type::Category::Array || baseType->category() == Type::Category::FixedBytes))
			{
				Json result = Json::object();
				result["kind"] = "array_length";
				result["base"] = exportExpr(memberAccess->expression());
				return result;
			}
		}

		// `.selector` on a function/method reference (e.g.
		// IERC20.transfer.selector, IERC721Receiver.onERC721Received.selector,
		// token.upgradeTo.selector). The base is a function reference whose
		// annotation carries the FunctionType / FunctionDefinition; lowering
		// that base to a *value* is neither meaningful nor possible (functions
		// are not values), so the generic `field` fallback below would
		// substitute a `{u256,0}` placeholder and silently lose the selector.
		// Instead emit explicit selector/signature/contract metadata taken
		// straight from the type annotations — the same FunctionType APIs the
		// contract-level exportForeignMethodSummary uses
		// (externalIdentifierHex / externalSignature). The OCaml frontend
		// consumes this as an opaque U256 selector literal; no name-based
		// recovery is performed here.
		if (memberAccess->memberName() == "selector")
		{
			Expression const& selBase = memberAccess->expression();
			FunctionType const* funType =
				dynamic_cast<FunctionType const*>(selBase.annotation().type);
			FunctionDefinition const* funcDef = nullptr;
			std::string methodName;
			if (auto const* baseMember = dynamic_cast<MemberAccess const*>(&selBase))
			{
				methodName = baseMember->memberName();
				funcDef = dynamic_cast<FunctionDefinition const*>(
					baseMember->annotation().referencedDeclaration);
			}
			if (!funcDef)
				if (auto const* baseIdent = dynamic_cast<Identifier const*>(&selBase))
					funcDef = dynamic_cast<FunctionDefinition const*>(
						baseIdent->annotation().referencedDeclaration);
			if (methodName.empty() && funcDef)
				methodName = funcDef->name();

			// Resolve a canonical external selector + signature: prefer the
			// annotation FunctionType (already the external interface function
			// type for a qualified member access); otherwise synthesize one from
			// the FunctionDefinition, exactly as the cross-contract resolver
			// (resolveStaticBaseCallTarget) and exportForeignMethodSummary do.
			std::string selectorHex;
			std::string signature;
			if (funType)
			{
				selectorHex = funType->externalIdentifierHex();
				signature = funType->externalSignature();
			}
			else if (funcDef)
			{
				selectorHex = funcDef->externalIdentifierHex();
				FunctionType ft(*funcDef);
				if (FunctionType const* iface = ft.interfaceFunctionType())
					signature = iface->externalSignature();
			}

			// Only treat this as a function selector when the type annotations
			// actually resolve to an external function. A struct/storage field
			// literally named "selector" has neither a FunctionType nor a
			// FunctionDefinition base and must fall through to the generic
			// `field` handling below unchanged.
			if (!selectorHex.empty())
			{
				Json result = Json::object();
				result["kind"] = "method_selector";
				result["method_name"] = methodName;
				if (signature.empty())
					result["method_signature"] = nullptr;
				else
					result["method_signature"] = signature;
				result["selector_hex"] = selectorHex;

				// contractId: prefer an explicit contract-qualified base
				// (IERC721Receiver.onERC721Received). null otherwise —
				// selector_hex and method_signature remain authoritative, so a
				// null contractId is never a guess.
				std::string contractId;
				if (auto const* baseMember = dynamic_cast<MemberAccess const*>(&selBase))
				{
					if (auto const* baseIdent =
							dynamic_cast<Identifier const*>(&baseMember->expression()))
					{
						if (auto const* cd = dynamic_cast<ContractDefinition const*>(
								baseIdent->annotation().referencedDeclaration))
							contractId = cd->name();
					}
				}
				if (contractId.empty())
					result["contractId"] = nullptr;
				else
					result["contractId"] = contractId;
				return result;
			}
		}
		Json result = Json::object();
		result["kind"] = "field";
		// [SolCore audit finding #9] No catch here: an unlowerable base
		// expression must propagate (to exportBody's catch-all, which turns
		// it into a diagnosable "unsupported_body" marker), not silently
		// become an untraceable `{u256,0}` -- the field access would then
		// read a real struct/tuple field off a fabricated zero base, exactly
		// the reintroduced `.selector`-class placeholder bug this file's
		// abi_encode*/generic-call sites were already hardened against.
		result["base"] = exportExpr(memberAccess->expression());
		result["field"] = memberAccess->memberName();
		return result;
	}

	if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr))
	{
		if (!indexAccess->indexExpression())
			throw UnsupportedSolCore("Index access without index is unsupported.");

		// Check if base type is an array or fixed bytes (not a mapping)
		Type const* baseType = indexAccess->baseExpression().annotation().type;
		if (baseType && (baseType->category() == Type::Category::Array ||
		                 baseType->category() == Type::Category::FixedBytes))
		{
			Json result = Json::object();
			result["kind"] = "array_get";
			result["base"] = exportExpr(indexAccess->baseExpression());
			result["index"] = exportExpr(*indexAccess->indexExpression());
			return result;
		}

		// Fall back to mapping index access
		try
		{
			auto [path, key] = exportStorageMapLValue(_expr);
			Json result = Json::object();
			result["kind"] = "storage_map_get";
			if (path.size() == 1)
				result["field"] = path.front();
			else
				result["path"] = jsonStringArray(path);
			result["key"] = key;
			return result;
		}
		catch (...)
		{
			// If mapping export fails, fall back to generic array_get
			Json result = Json::object();
			result["kind"] = "array_get";
			result["base"] = exportExpr(indexAccess->baseExpression());
			result["index"] = exportExpr(*indexAccess->indexExpression());
			return result;
		}
	}

	// Array/bytes slicing: arr[start:end], arr[:end], arr[start:]
	if (auto const* rangeAccess = dynamic_cast<IndexRangeAccess const*>(&_expr))
	{
		Json result = Json::object();
		result["kind"] = "internal_call";
		result["function"] = "array_slice";
		result["args"] = Json::array();
		result["args"].emplace_back(exportExpr(rangeAccess->baseExpression()));
		if (rangeAccess->startExpression())
			result["args"].emplace_back(exportExpr(*rangeAccess->startExpression()));
		else
		{
			Json zero = Json::object();
			zero["kind"] = "u256";
			zero["value"] = "0";
			result["args"].emplace_back(zero);
		}
		if (rangeAccess->endExpression())
			result["args"].emplace_back(exportExpr(*rangeAccess->endExpression()));
		else
		{
			// End is array length
			Json len = Json::object();
			len["kind"] = "array_length";
			len["base"] = exportExpr(rangeAccess->baseExpression());
			result["args"].emplace_back(len);
		}
		return result;
	}

	if (auto const* binary = dynamic_cast<BinaryOperation const*>(&_expr))
	{
		Json result = Json::object();

		// SolCore audit finding #5: `<, <=, >, >=, /, %, >>` have different
		// bit patterns/results for signed vs. unsigned operands (e.g.
		// int256(-1) < int256(1) is true under SLT but u256_lt(2^256-1, 1)
		// is false), yet this lowering used to route every one of these
		// operators to its unsigned EVM-node counterpart with no signedness
		// marker at all. `commonType` (set by the type checker on the
		// `BinaryOperation` itself) is the actual operand type used for the
		// operation -- not necessarily the expression's result type, which
		// for comparisons is `bool` -- so it is the right thing to consult
		// here, except for the shift operators: Solidity always types the
		// shift *amount* (rhs) as unsigned regardless of the value being
		// shifted, so `commonType` doesn't reflect the signedness that
		// matters for `>>`; the left operand's own type does.
		//
		// There is no signed counterpart of `u256_div`/`u256_mod`/`u256_lt`/
		// `u256_le`/`u256_gt`/`u256_ge`/`u256_shr` that the OCaml frontend
		// and Lean runtime can consume yet (seeing one of these `kind`s
		// with a signed operand would previously produce a false, silently
		// wrong preservation certificate downstream). Until that signed
		// vocabulary exists end-to-end, fail closed here -- via the same
		// `UnsupportedSolCore` -> `unsupported_body` mechanism used
		// throughout this file -- instead of manufacturing a certificate
		// over the wrong semantics.
		auto isSignedIntegerType = [](Type const* _type) -> bool {
			if (!_type || _type->category() != Type::Category::Integer)
				return false;
			auto const* intType = dynamic_cast<IntegerType const*>(_type);
			return intType && intType->isSigned();
		};

		switch (binary->getOperator())
		{
		case Token::Add:
			result["kind"] = "u256_add";
			markUncheckedContext(result);
			tagNarrowArithWidth(result, binary->annotation().commonType);
			break;
		case Token::Sub:
			result["kind"] = "u256_sub";
			markUncheckedContext(result);
			tagNarrowArithWidth(result, binary->annotation().commonType);
			break;
		case Token::Mul:
			result["kind"] = "u256_mul";
			markUncheckedContext(result);
			tagNarrowArithWidth(result, binary->annotation().commonType);
			break;
		case Token::Div:
			if (isSignedIntegerType(binary->annotation().commonType))
				throw UnsupportedSolCore(
					"Signed division ('/' on a signed integer type) has no "
					"faithful SolCore lowering yet; the unsigned u256_div "
					"primitive would silently produce the wrong result "
					"whenever an operand is negative.");
			result["kind"] = "u256_div";
			break;
		case Token::Mod:
			if (isSignedIntegerType(binary->annotation().commonType))
				throw UnsupportedSolCore(
					"Signed modulo ('%' on a signed integer type) has no "
					"faithful SolCore lowering yet; the unsigned u256_mod "
					"primitive would silently produce the wrong result "
					"whenever an operand is negative.");
			result["kind"] = "u256_mod";
			break;
		case Token::Exp:
			result["kind"] = "u256_exp";
			break;
		case Token::Equal:
			result["kind"] = "u256_eq";
			break;
		case Token::NotEqual:
			result["kind"] = "u256_ne";
			break;
		case Token::LessThan:
			if (isSignedIntegerType(binary->annotation().commonType))
				throw UnsupportedSolCore(
					"Signed comparison ('<' on a signed integer type) has "
					"no faithful SolCore lowering yet; the unsigned "
					"u256_lt primitive silently flips the result whenever "
					"an operand is negative.");
			result["kind"] = "u256_lt";
			break;
		case Token::LessThanOrEqual:
			if (isSignedIntegerType(binary->annotation().commonType))
				throw UnsupportedSolCore(
					"Signed comparison ('<=' on a signed integer type) has "
					"no faithful SolCore lowering yet; the unsigned "
					"u256_le primitive silently flips the result whenever "
					"an operand is negative.");
			result["kind"] = "u256_le";
			break;
		case Token::GreaterThan:
			if (isSignedIntegerType(binary->annotation().commonType))
				throw UnsupportedSolCore(
					"Signed comparison ('>' on a signed integer type) has "
					"no faithful SolCore lowering yet; the unsigned "
					"u256_gt primitive silently flips the result whenever "
					"an operand is negative.");
			result["kind"] = "u256_gt";
			break;
		case Token::GreaterThanOrEqual:
			if (isSignedIntegerType(binary->annotation().commonType))
				throw UnsupportedSolCore(
					"Signed comparison ('>=' on a signed integer type) has "
					"no faithful SolCore lowering yet; the unsigned "
					"u256_ge primitive silently flips the result whenever "
					"an operand is negative.");
			result["kind"] = "u256_ge";
			break;
		case Token::And:
			result["kind"] = "bool_and";
			break;
		case Token::Or:
			result["kind"] = "bool_or";
			break;
		case Token::BitAnd:
			result["kind"] = "u256_bitand";
			break;
		case Token::BitOr:
			result["kind"] = "u256_bitor";
			break;
		case Token::BitXor:
			result["kind"] = "u256_bitxor";
			break;
		case Token::SHL:
			result["kind"] = "u256_shl";
			break;
		case Token::SAR:
			if (isSignedIntegerType(binary->leftExpression().annotation().type))
				throw UnsupportedSolCore(
					"Signed right shift ('>>' on a signed integer type) "
					"has no faithful SolCore lowering yet; the logical-"
					"shift u256_shr primitive silently drops sign "
					"extension for negative values.");
			result["kind"] = "u256_shr";
			break;
		default:
			throw UnsupportedSolCore("Unsupported binary operator in SolCore exporter.");
		}
		result["lhs"] = exportExpr(binary->leftExpression());
		result["rhs"] = exportExpr(binary->rightExpression());
		return result;
	}

	if (auto const* unary = dynamic_cast<UnaryOperation const*>(&_expr))
	{
		Json result = Json::object();
		switch (unary->getOperator())
		{
		case Token::Not:
			result["kind"] = "bool_not";
			result["operand"] = exportExpr(unary->subExpression());
			return result;
		case Token::BitNot:
			result["kind"] = "u256_bitnot";
			result["operand"] = exportExpr(unary->subExpression());
			return result;
		case Token::Inc:
		{
			result["kind"] = "u256_add";
			markUncheckedContext(result);
			tagNarrowArithWidth(result, unary->subExpression().annotation().type);
			result["lhs"] = exportExpr(unary->subExpression());
			Json one = Json::object();
			one["kind"] = "u256";
			one["value"] = "1";
			result["rhs"] = one;
			return result;
		}
		case Token::Dec:
		{
			result["kind"] = "u256_sub";
			markUncheckedContext(result);
			tagNarrowArithWidth(result, unary->subExpression().annotation().type);
			result["lhs"] = exportExpr(unary->subExpression());
			Json one = Json::object();
			one["kind"] = "u256";
			one["value"] = "1";
			result["rhs"] = one;
			return result;
		}
		case Token::Sub:
		{
			result["kind"] = "u256_sub";
			markUncheckedContext(result);
			Json zero = Json::object();
			zero["kind"] = "u256";
			zero["value"] = "0";
			result["lhs"] = zero;
			result["rhs"] = exportExpr(unary->subExpression());
			return result;
		}
		default:
			throw UnsupportedSolCore("Unsupported unary operator in SolCore exporter.");
		}
	}

	if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
	{
		Json result = Json::object();
		result["kind"] = "conditional";
		result["cond"] = exportExpr(conditional->condition());
		result["true_value"] = exportExpr(conditional->trueExpression());
		result["false_value"] = exportExpr(conditional->falseExpression());
		return result;
	}

	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expr))
	{
		// If single-element tuple, unwrap it
		if (tuple->components().size() == 1 && tuple->components().front())
			return exportExpr(*tuple->components().front());
		// Multi-element tuples: export as tuple
		Json result = Json::object();
		result["kind"] = "tuple";
		result["elements"] = Json::array();
		for (auto const& component: tuple->components())
		{
			if (component)
				result["elements"].emplace_back(exportExpr(*component));
			else
				result["elements"].emplace_back(Json());
		}
		return result;
	}

	if (auto const* assignment = dynamic_cast<Assignment const*>(&_expr))
	{
		// Assignment as expression (e.g. `while((y = x) > 1)`)
		if (auto const* identifier = dynamic_cast<Identifier const*>(&assignment->leftHandSide()))
		{
			auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
			Json result = Json::object();
			result["kind"] = "assign_expr";
			result["name"] = (decl && !decl->name().empty()) ? decl->name() : identifier->name();
			result["value"] = exportExpr(assignment->rightHandSide());
			return result;
		}
		// Non-identifier LHS: index access, member access, etc. — export as generic assign_expr
		{
			Json result = Json::object();
			result["kind"] = "assign_expr";
			result["target"] = exportExpr(assignment->leftHandSide());
			result["value"] = exportExpr(assignment->rightHandSide());
			return result;
		}
	}

	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
	{
		// Type conversions (e.g. uint256(x), address(x)) are no-ops at EVM
		// level for every case EXCEPT a genuine narrowing integer downcast
		// (e.g. `uint128(x)` from a `uint256` — the SafeCast/explicit-downcast
		// idiom SOLCORE_MATH_BUG_CLASSES_PLAN.md §3.5/§6.3 (Task 11) flags:
		// before this, the target bit-width carried nowhere through this
		// exporter, so `SolCoreOfJson.ml` had no way to model the truncation
		// and silently lowered the downcast to an untruncated copy — making
		// `require(downcasted == value)`-style validation checks
		// tautological under the old model). Widening/same-width/non-integer
		// conversions keep the exact old pass-through behavior (this patch
		// changes nothing for them), so only the previously-mismodeled case
		// changes.
		if (*call->annotation().kind == FunctionCallKind::TypeConversion)
		{
			if (call->arguments().empty())
				throw UnsupportedSolCore("Empty type conversion in SolCore exporter.");
			Expression const& argExpr = *call->arguments().front();
			Json innerJson = exportExpr(argExpr);
			Type const* targetType = call->annotation().type;
			Type const* sourceType = argExpr.annotation().type;
			if (
				targetType && sourceType &&
				targetType->category() == Type::Category::Integer &&
				sourceType->category() == Type::Category::Integer
			)
			{
				auto const* targetInt = dynamic_cast<IntegerType const*>(targetType);
				auto const* sourceInt = dynamic_cast<IntegerType const*>(sourceType);
				if (targetInt && sourceInt && targetInt->numBits() < sourceInt->numBits())
				{
					Json result = Json::object();
					result["kind"] = "truncate";
					result["target_bits"] = static_cast<int>(targetInt->numBits());
					result["value"] = innerJson;
					return result;
				}
			}
			return innerJson;
		}

		// Struct constructor calls: S(field1, field2, ...)
		if (*call->annotation().kind == FunctionCallKind::StructConstructorCall)
		{
			Json result = Json::object();
			result["kind"] = "struct_constructor";
			// Try to get the struct name from the expression
			if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
				result["name"] = callee->name();
			else if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
				result["name"] = memberAccess->memberName();
			else
				result["name"] = "Unknown";
			result["args"] = Json::array();
			// [SolCore audit finding #9] No catch-and-substitute-0 here: an
			// unlowerable struct-constructor argument must propagate to
			// exportBody's catch-all rather than silently becoming an
			// untraceable `{u256,0}` field value.
			for (auto const& arg: call->arguments())
				result["args"].emplace_back(exportExpr(*arg));
			return result;
		}

		// Handle new expressions: new bytes(n), new string(n), new uint256[](n)
		if (auto const* newExpr = dynamic_cast<NewExpression const*>(&call->expression()))
		{
			// Check if this is an empty array allocation: new T[](0)
			// In that case, emit unit which the code generator maps to default = []
			if (call->arguments().size() == 1)
			{
				auto const* sizeArg = dynamic_cast<Literal const*>(call->arguments().front().get());
				if (sizeArg && sizeArg->value() == "0")
				{
					Json result = Json::object();
					result["kind"] = "unit";
					return result;
				}
			}
			Json result = Json::object();
			result["kind"] = "internal_call";
			result["function"] = "new_array";
			result["args"] = Json::array();
			// The argument is the size.
			// [SolCore audit finding #9] No catch-and-substitute-0 here: an
			// unlowerable size argument must propagate to exportBody's
			// catch-all rather than silently allocating an untraceable
			// zero-sized/zero-length array.
			for (auto const& arg: call->arguments())
				result["args"].emplace_back(exportExpr(*arg));
			// Also include the type information
			try
			{
				result["element_type"] = exportTypeName(newExpr->typeName());
			}
			catch (...)
			{
				result["element_type"] = Json("u8");
			}
			return result;
		}

		// Check for builtin function calls by identifier name
		if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
		{
			// assert(cond)
			if (callee->name() == "assert")
			{
				if (call->arguments().empty())
					throw UnsupportedSolCore("assert() without arguments is unsupported.");
				Json result = Json::object();
				result["kind"] = "assert";
				result["cond"] = exportExpr(*call->arguments().front());
				return result;
			}

			// addmod(x, y, m)
			if (callee->name() == "addmod")
			{
				if (call->arguments().size() < 3)
					throw UnsupportedSolCore("addmod() requires 3 arguments.");
				Json result = Json::object();
				result["kind"] = "u256_addmod";
				result["x"] = exportExpr(*call->arguments()[0]);
				result["y"] = exportExpr(*call->arguments()[1]);
				result["m"] = exportExpr(*call->arguments()[2]);
				return result;
			}

			// mulmod(x, y, m)
			if (callee->name() == "mulmod")
			{
				if (call->arguments().size() < 3)
					throw UnsupportedSolCore("mulmod() requires 3 arguments.");
				Json result = Json::object();
				result["kind"] = "u256_mulmod";
				result["x"] = exportExpr(*call->arguments()[0]);
				result["y"] = exportExpr(*call->arguments()[1]);
				result["m"] = exportExpr(*call->arguments()[2]);
				return result;
			}

			// sha256(data), keccak256(data), ecrecover(h,v,r,s), gasleft()
			if (callee->name() == "sha256" || callee->name() == "keccak256" ||
			    callee->name() == "ecrecover" || callee->name() == "gasleft" ||
			    callee->name() == "blockhash")
			{
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = callee->name();
				result["args"] = Json::array();
				// [SolCore audit finding #9] No catch-and-substitute-0 here:
				// an unlowerable hash/ecrecover/gasleft argument must
				// propagate to exportBody's catch-all rather than silently
				// hashing/recovering over an untraceable zero.
				for (auto const& arg: call->arguments())
					result["args"].emplace_back(exportExpr(*arg));
				return result;
			}

			// require(cond) / require(cond, msg) — when used as expression
			if (callee->name() == "require")
			{
				if (!call->arguments().empty())
				{
					Json result = Json::object();
					result["kind"] = "assert";
					result["cond"] = exportExpr(*call->arguments().front());
					return result;
				}
			}

			// Internal function call: callee references a FunctionDefinition.
			// Plain-identifier calls are VIRTUAL dispatch — name the slot
			// winner, not the lexically referenced declaration (see
			// virtualCallTargetName).
			auto const* funcDef = dynamic_cast<FunctionDefinition const*>(callee->annotation().referencedDeclaration);
			if (funcDef)
			{
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = virtualCallTargetName(*funcDef);
				result["args"] = Json::array();
				for (auto const& arg: call->arguments())
					result["args"].emplace_back(exportExpr(*arg));
				return result;
			}
		}

		// Check for member calls on arrays and contracts
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
		{
			// Statically-bound super/base-qualified internal calls:
			// `super.f(...)` and `Base.f(...)`. Both must lower to the
			// flattened `f__super__<DefiningContract>` sibling (emitted by the
			// super-referenced-functions pass in exportContract) — NEVER to
			// the bare virtual-slot name, which for `Base.f()` inside an
			// override of `f` produced a self-forwarding body that silently
			// skipped `Base.f`'s refinements (e.g. Governance.state →
			// GovernorTimelockControl.state's timelock-queue logic).
			if (auto resolved = resolveStaticBaseCallTarget(*call, *memberAccess, activeExportContract))
			{
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = flattenedStaticBaseCallName(*resolved);
				result["args"] = Json::array();
				for (auto const& arg: call->arguments())
					result["args"].emplace_back(exportExpr(*arg));
				return result;
			}

			// OpenZeppelin SafeCast narrowing downcasts (`x.toUint128()` via
			// `using SafeCast for uint256`, or the qualified
			// `SafeCast.toUint128(x)` form) — see
			// isKnownOzSafeCastNarrowingDowncast's docstring above. Must run
			// before the using-for/qualified-call `internal_call` branches
			// further below, which would otherwise catch this same call
			// shape first and export it as an opaque call.
			if (auto const* safeCastFuncDef = dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
			{
				int targetBits = 0;
				if (isKnownOzSafeCastNarrowingDowncast(*safeCastFuncDef, targetBits))
				{
					Type const* receiverType = memberAccess->expression().annotation().type;
					bool isUsingForCall =
						receiverType &&
						receiverType->category() != Type::Category::TypeType &&
						receiverType->category() != Type::Category::Module;
					Expression const* argExpr = nullptr;
					if (isUsingForCall)
						argExpr = &memberAccess->expression();
					else if (!call->arguments().empty())
						argExpr = call->arguments().front().get();
					if (argExpr)
					{
						Json result = Json::object();
						result["kind"] = "truncate";
						result["target_bits"] = targetBits;
						result["value"] = exportExpr(*argExpr);
						return result;
					}
					// Couldn't resolve the argument (shouldn't happen for any
					// real SafeCast call site) — fail closed and fall through
					// to the generic internal_call path below rather than
					// guessing.
				}
			}

			// abi.encode, abi.encodePacked, abi.decode, abi.encodeWithSelector, abi.encodeWithSignature
			{
				Type const* baseType = memberAccess->expression().annotation().type;
				if (baseType && baseType->category() == Type::Category::Magic)
				{
					auto const* magicType = dynamic_cast<MagicType const*>(baseType);
					if (magicType && magicType->kind() == MagicType::Kind::ABI)
					{
						// abi.decode(data, (T1, T2, ...)): the second argument is a
						// syntactic *type list*, not a value expression. The parser
						// always wraps a parenthesized argument list in a
						// TupleExpression (even a single-element one — see
						// Parser::parsePrimaryExpression's "(x) is not a real tuple"
						// comment, which only affects later *type* checking, not the
						// AST shape), whose components are ElementaryTypeNameExpression
						// nodes referring to a type, not a value. Recursing into it via
						// the generic exportExpr args-loop below would fall through every
						// dynamic_cast in exportExpr (ElementaryTypeNameExpression isn't
						// one of the handled Expression subtypes) and hit the ultimate
						// fallback — which used to silently substitute a `0`
						// u256/`_unsupported_expr` sentinel in place of the type list, and
						// now throws instead (see that fallback's comment below). Either
						// way, routing the type list through exportExpr both mis-describes
						// a type as a value and — more importantly — throws away
						// information the OCaml side needs to synthesize a genuinely
						// opaque runtime value of the correct decoded type(s). So export
						// the type list out-of-band as canonical Solidity type-name
						// strings (the same `Type::toString()` used for signature-suffix
						// disambiguation above) via a sibling `decode_types` field
						// instead of routing it through exportExpr.
						if (memberAccess->memberName() == "decode" && call->arguments().size() == 2)
						{
							auto const* typesTuple = dynamic_cast<TupleExpression const*>(call->arguments()[1].get());
							if (!typesTuple)
								throw UnsupportedSolCore("abi.decode: expected a parenthesized type list as the second argument.");

							Json result = Json::object();
							result["kind"] = "internal_call";
							result["function"] = "abi_decode";
							result["args"] = Json::array();
							result["args"].emplace_back(exportExpr(*call->arguments()[0]));

							Json decodeTypes = Json::array();
							for (auto const& component: typesTuple->components())
							{
								if (!component)
									throw UnsupportedSolCore("abi.decode: omitted entry in type list is unsupported.");
								Type const* componentType = component->annotation().type;
								auto const* typeType = dynamic_cast<TypeType const*>(componentType);
								if (!typeType || !typeType->actualType())
									throw UnsupportedSolCore("abi.decode: type-list entry is not a resolvable type name.");
								// _withoutDataLocation=true: the type-list entries are bare
								// type names (no explicit storage/memory/calldata location
								// was written), so Type::toString(false)'s default location
								// suffix (e.g. "bytes storage pointer") would produce a
								// string the OCaml-side sc_ty_of_external_signature_type
								// parser doesn't recognize. Canonical form (e.g. "bytes",
								// "address", "uint256[]") is what that parser expects.
								decodeTypes.emplace_back(typeType->actualType()->toString(true));
							}
							result["decode_types"] = decodeTypes;
							return result;
						}

						Json result = Json::object();
						result["kind"] = "internal_call";
						result["function"] = "abi_" + memberAccess->memberName();
						result["args"] = Json::array();
						for (auto const& arg: call->arguments())
							// No catch here: an argument this pipeline can't lower must
							// propagate (to exportBody's catch-all, which turns it into a
							// diagnosable "unsupported_body" marker), not silently become
							// a bare `0` with no trace at all. abi.decode itself never
							// reaches this loop (handled above, out-of-band); this is only
							// abi.encode/abi.encodePacked/abi.encodeWithSignature now.
							result["args"].emplace_back(exportExpr(*arg));
						return result;
					}
				}
			}

			// Array pop: arr.pop()
			if (memberAccess->memberName() == "pop")
			{
				Type const* baseType = memberAccess->expression().annotation().type;
				if (baseType && baseType->category() == Type::Category::Array)
				{
					Json result = Json::object();
					result["kind"] = "array_pop";
					result["base"] = exportExpr(memberAccess->expression());
					return result;
				}
			}

			// Array push as expression: arr.push(value) returning reference
			if (memberAccess->memberName() == "push")
			{
				Type const* baseType = memberAccess->expression().annotation().type;
				if (baseType && baseType->category() == Type::Category::Array)
				{
					Json result = Json::object();
					result["kind"] = "array_push";
					result["base"] = exportExpr(memberAccess->expression());
					if (!call->arguments().empty())
						result["value"] = exportExpr(*call->arguments().front());
					else
					{
						Json defaultVal = Json::object();
						defaultVal["kind"] = "u256";
						defaultVal["value"] = "0";
						result["value"] = defaultVal;
					}
					return result;
				}
			}

			// using-for library extension call: receiver.method(args) resolves
			// to an internal library function whose receiver is the first
			// parameter. Type/module-qualified L.f(args) calls are handled
			// below and do not get a receiver argument.
			if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
			{
				Type const* baseType = memberAccess->expression().annotation().type;
				if (
					baseType &&
					baseType->category() != Type::Category::TypeType &&
					baseType->category() != Type::Category::Module &&
					funcDef->visibility() <= Visibility::Internal
				)
				{
					Json result = Json::object();
					result["kind"] = "internal_call";
					result["function"] = exportedFunctionName(*funcDef);
					addInternalLibraryCallContractId(result, *funcDef);
					result["args"] = Json::array();
					result["args"].emplace_back(exportExpr(memberAccess->expression()));
					for (auto const& arg: call->arguments())
						result["args"].emplace_back(exportExpr(*arg));
					return result;
				}
			}

			// External contract call: contract.method(args)
			Type const* baseType = memberAccess->expression().annotation().type;
			if (baseType && baseType->category() == Type::Category::Contract)
			{
				if (auto externalCall = exportExternalContractCall(*call, *memberAccess, false))
					return *externalCall;
			}

			// Library-qualified or type-qualified function call: L.f(args)
			// The member access resolves to a FunctionDefinition when calling through a library/type namespace
			if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
			{
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = exportedFunctionName(*funcDef);
				addInternalLibraryCallContractId(result, *funcDef);
				result["args"] = Json::array();
				for (auto const& arg: call->arguments())
					result["args"].emplace_back(exportExpr(*arg));
				return result;
			}
		}

		// Check for low-level calls: address.call{value: ...}(data) and address.delegatecall(data)
		// These appear as FunctionCall whose expression is FunctionCallOptions
		if (auto const* options = dynamic_cast<FunctionCallOptions const*>(&call->expression()))
		{
			if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&options->expression()))
			{
				if (auto externalCall = exportExternalContractCall(*call, *memberAccess, false))
					return *externalCall;

				if (memberAccess->memberName() == "call" || memberAccess->memberName() == "delegatecall")
				{
					Json result = Json::object();
					result["kind"] = "low_level_call";
					result["callKind"] = lowLevelCallKindString(memberAccess->memberName());
					result["target"] = exportExpr(memberAccess->expression());

					// Extract value from options
					Json valueExpr = Json::object();
					valueExpr["kind"] = "u256";
					valueExpr["value"] = "0";
					for (size_t i = 0; i < options->names().size(); ++i)
					{
						if (*options->names()[i] == "value")
						{
							valueExpr = exportExpr(*options->options()[i]);
							break;
						}
					}
					result["value"] = valueExpr;

					// Data is the first argument to the call
					if (!call->arguments().empty())
						result["data"] = exportExpr(*call->arguments().front());
					else
					{
						Json emptyData = Json::object();
						emptyData["kind"] = "string";
						emptyData["value"] = "";
						result["data"] = emptyData;
					}
					return result;
				}
			}
		}

		// Check for low-level calls without options: address.call(data) or address.delegatecall(data)
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
		{
			if (memberAccess->memberName() == "call" || memberAccess->memberName() == "delegatecall")
			{
				Type const* baseType = memberAccess->expression().annotation().type;
				if (baseType && baseType->category() == Type::Category::Address)
				{
					Json result = Json::object();
					result["kind"] = "low_level_call";
					result["callKind"] = lowLevelCallKindString(memberAccess->memberName());
					result["target"] = exportExpr(memberAccess->expression());

					Json valueExpr = Json::object();
					valueExpr["kind"] = "u256";
					valueExpr["value"] = "0";
					result["value"] = valueExpr;

					if (!call->arguments().empty())
						result["data"] = exportExpr(*call->arguments().front());
					else
					{
						Json emptyData = Json::object();
						emptyData["kind"] = "string";
						emptyData["value"] = "";
						result["data"] = emptyData;
					}
					return result;
				}
			}
		}
	}

	// Fallback for FunctionCall that doesn't match known patterns
	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
	{
		// Generic function call — export as internal_call with best-effort name
		Json result = Json::object();
		result["kind"] = "internal_call";
		if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
			result["function"] = callee->name();
		else if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
		{
			if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
				result["function"] = exportedFunctionName(*funcDef);
			else
				result["function"] = memberAccess->memberName();
		}
		else
			result["function"] = "unknown_call";
		result["args"] = Json::array();
		for (auto const& arg: call->arguments())
			// No catch here — see the abi_encode* loop above for the rationale:
			// an unlowerable argument must propagate to exportBody's catch-all
			// rather than silently becoming an untraceable `0`.
			result["args"].emplace_back(exportExpr(*arg));
		return result;
	}

	// Fallback for FunctionCallOptions (e.g., addr.call{value: 1}): export as its inner expression
	if (auto const* options = dynamic_cast<FunctionCallOptions const*>(&_expr))
		return exportExpr(options->expression());

	// Ultimate fallback: every expression form this exporter knows how to
	// recognize is handled above (Identifier, IndexAccess, Literal,
	// MemberAccess, IndexRangeAccess, BinaryOperation, UnaryOperation,
	// Conditional, TupleExpression, Assignment, FunctionCall,
	// FunctionCallOptions). Reaching here means `_expr` is a genuinely
	// unrecognized Expression subtype (e.g. ElementaryTypeNameExpression used
	// outside abi.decode's dedicated handling above). Throwing — instead of
	// the old `{"kind":"u256","value":"0","_unsupported_expr":true}`
	// placeholder — lets exportBody's catch-all turn this into a diagnosable
	// "unsupported_body" marker with a real reason string, rather than
	// manufacturing a fake value that some downstream consumer has to
	// remember to specifically check for.
	throw UnsupportedSolCore(
		"exportExpr: unrecognized expression node (no matching Expression subtype)");
}

Json exportRevertPayload(FunctionCall const& _call)
{
	auto exportCustomError = [&](ErrorDefinition const& _error) {
		Json result = Json::object();
		result["error"] = Json::object();
		result["error"]["name"] = _error.name();
		try
		{
			FunctionType errorType(_error);
			result["error"]["signature"] = errorType.externalSignature();
		}
		catch (...)
		{
		}
		Json argTypes = Json::array();
		bool argTypesOk = true;
		for (auto const& parameter: _error.parameters())
		{
			try
			{
				argTypes.emplace_back(exportTypeName(parameter->typeName()));
			}
			catch (...)
			{
				argTypesOk = false;
				break;
			}
		}
		if (argTypesOk && !argTypes.empty())
			result["error"]["argTypes"] = std::move(argTypes);
		Json args = Json::array();
		for (auto const& argument: _call.arguments())
			args.emplace_back(exportExpr(*argument));
		if (!args.empty())
			result["error"]["args"] = std::move(args);
		if (_call.arguments().size() == 1)
		{
			if (auto const* message = dynamic_cast<Literal const*>(_call.arguments().front().get()))
				if (message->token() == Token::StringLiteral)
					result["error"]["data"] = message->value();
		}
		return result;
	};

	Json result = Json::object();
	Declaration const* calleeDecl = nullptr;
	if (auto const* callee = dynamic_cast<Identifier const*>(&_call.expression()))
		calleeDecl = callee->annotation().referencedDeclaration;
	else if (auto const* callee = dynamic_cast<MemberAccess const*>(&_call.expression()))
		calleeDecl = callee->annotation().referencedDeclaration;
	if (auto const* errorDef = dynamic_cast<ErrorDefinition const*>(calleeDecl))
		return exportCustomError(*errorDef);

	// Given the single argument expression that carries the revert reason
	// (a string literal message, a custom-error constructor call, or some
	// other non-literal expression), export it into `result` as
	// "error"/"message"/"payload" exactly as before. Factored out so it can
	// be applied at the correct argument INDEX for each builtin below --
	// `require(cond, reason)` carries its reason at index 1 (index 0 is the
	// condition), while the global `revert(reason)` builtin carries its
	// (sole, optional) reason at index 0. Reusing one fixed index for both
	// (the previous behavior) silently dropped every `revert(reason)` call's
	// message, since `revert` never has a 2nd argument to find it at.
	auto exportReasonArg = [&](Expression const& _reasonArg) -> Json {
		if (auto const* errorCall = dynamic_cast<FunctionCall const*>(&_reasonArg))
		{
			if (auto const* errorDef =
					dynamic_cast<ErrorDefinition const*>(
						dynamic_cast<Identifier const*>(&errorCall->expression()) ?
							dynamic_cast<Identifier const*>(&errorCall->expression())->annotation().referencedDeclaration :
						dynamic_cast<MemberAccess const*>(&errorCall->expression()) ?
							dynamic_cast<MemberAccess const*>(&errorCall->expression())->annotation().referencedDeclaration :
							nullptr))
				return exportCustomError(*errorDef);
		}
		auto const* message = dynamic_cast<Literal const*>(&_reasonArg);
		if (message && message->token() == Token::StringLiteral)
		{
			Json reasonResult = Json::object();
			reasonResult["message"] = message->value();
			return reasonResult;
		}
		// Non-string revert message (e.g. custom error): export as generic expression
		Json reasonResult = Json::object();
		try
		{
			reasonResult["payload"] = exportExpr(_reasonArg);
		}
		catch (...)
		{
			reasonResult["message"] = "(non-string revert payload)";
		}
		return reasonResult;
	};

	if (auto const* callee = dynamic_cast<Identifier const*>(&_call.expression()))
	{
		// `require(cond)` / `require(cond, reason)`: the reason, if present,
		// is always the 2nd argument (index 1) -- index 0 is the condition.
		if (callee->name() == "require" && _call.arguments().size() >= 2)
			return exportReasonArg(*_call.arguments().at(1));

		// The global `revert()` / `revert(reason)` builtin: unlike `require`,
		// it takes no leading condition, so its (sole, optional) reason -- if
		// the caller passed one -- is the 1st argument (index 0).
		if (callee->name() == "revert" && _call.arguments().size() >= 1)
			return exportReasonArg(*_call.arguments().at(0));
	}

	// Fallback: return empty payload rather than throwing (genuinely
	// argument-less `require(cond)` / `revert()`, or an unrecognized callee).
	return result;
}

Json exportStmt(Statement const& _stmt);

Json replaceModifierPlaceholders(Json const& _stmt, Json const& _replacementBody)
{
	if (!_stmt.is_object())
		return _stmt;

	std::string kind = _stmt.value("kind", ""s);
	if (kind == "placeholder")
		return _replacementBody;

	Json result = _stmt;
	if (kind == "block")
	{
		Json statements = Json::array();
		for (auto const& stmt: _stmt["statements"])
			statements.emplace_back(replaceModifierPlaceholders(stmt, _replacementBody));
		result["statements"] = std::move(statements);
	}
	else if (kind == "if")
	{
		result["then"] = replaceModifierPlaceholders(_stmt["then"], _replacementBody);
		if (_stmt.contains("else") && !_stmt["else"].is_null())
			result["else"] = replaceModifierPlaceholders(_stmt["else"], _replacementBody);
	}
	else if (kind == "while" || kind == "do_while")
		result["body"] = replaceModifierPlaceholders(_stmt["body"], _replacementBody);
	else if (kind == "for")
	{
		if (_stmt.contains("init") && !_stmt["init"].is_null())
			result["init"] = replaceModifierPlaceholders(_stmt["init"], _replacementBody);
		if (_stmt.contains("post") && !_stmt["post"].is_null())
			result["post"] = replaceModifierPlaceholders(_stmt["post"], _replacementBody);
		result["body"] = replaceModifierPlaceholders(_stmt["body"], _replacementBody);
	}
	else if (kind == "try_catch")
	{
		Json clauses = Json::array();
		for (auto const& clause: _stmt["clauses"])
		{
			Json newClause = clause;
			newClause["body"] = replaceModifierPlaceholders(clause["body"], _replacementBody);
			clauses.emplace_back(std::move(newClause));
		}
		result["clauses"] = std::move(clauses);
	}

	return result;
}

ModifierDefinition const* resolveModifierDefinition(
	FunctionDefinition const& _function,
	ModifierInvocation const& _modifierInvocation)
{
	auto modifierDefinition = dynamic_cast<ModifierDefinition const*>(
		_modifierInvocation.name().annotation().referencedDeclaration
	);
	if (!modifierDefinition)
		return nullptr;

	if (_function.isFree())
		return modifierDefinition;

	ContractDefinition const* contract = _function.annotation().contract;
	if (!contract)
		return modifierDefinition;

	if (
		_modifierInvocation.name().annotation().requiredLookup.set() &&
		*_modifierInvocation.name().annotation().requiredLookup == VirtualLookup::Virtual
	)
		return &modifierDefinition->resolveVirtual(*contract);

	return modifierDefinition;
}

Json modifierParameterBindings(
	ModifierDefinition const& _modifierDefinition,
	std::vector<ASTPointer<Expression>> const* _arguments)
{
	Json bindings = Json::array();
	auto const& params = _modifierDefinition.parameters();
	size_t argCount = _arguments ? _arguments->size() : 0;
	for (size_t i = 0; i < params.size(); ++i)
	{
		VariableDeclaration const& param = *params[i];
		Json letStmt = Json::object();
		letStmt["kind"] = "let";
		letStmt["name"] = param.name().empty() ? ("modifier_arg" + std::to_string(i)) : param.name();
		try
		{
			letStmt["type"] = exportTypeName(param.typeName());
		}
		catch (...)
		{
			letStmt["type"] = Json("u256");
		}
		if (_arguments && i < argCount)
			letStmt["value"] = exportExpr(*(*_arguments)[i]);
		else
			letStmt["value"] = defaultValueForTypeName(param.typeName());
		bindings.emplace_back(std::move(letStmt));
	}
	return bindings;
}

Json expandModifiers(FunctionDefinition const& _function, Json _body)
{
	for (auto it = _function.modifiers().rbegin(); it != _function.modifiers().rend(); ++it)
	{
		ModifierInvocation const& modifierInvocation = *it->get();
		ModifierDefinition const* modifierDefinition = resolveModifierDefinition(_function, modifierInvocation);
		if (!modifierDefinition || !modifierDefinition->isImplemented())
			continue;

		Json modifierBody = exportStmt(modifierDefinition->body());
		if (!modifierBody.is_object() || modifierBody.value("kind", ""s) != "block")
			continue;

		Json expandedModifier = replaceModifierPlaceholders(modifierBody, _body);
		Json statements = Json::array();
		for (auto const& binding: modifierParameterBindings(*modifierDefinition, modifierInvocation.arguments()))
			statements.emplace_back(binding);
		for (auto const& stmt: expandedModifier["statements"])
			statements.emplace_back(stmt);
		expandedModifier["statements"] = std::move(statements);
		_body = std::move(expandedModifier);
	}

	return _body;
}

Json exportAssignment(Expression const& _lhs, Token _op, Expression const& _rhs)
{
	auto mkDirectStorage = [&](std::string const& _field, Json const& _value) {
		Json result = Json::object();
		result["kind"] = "storage_set";
		result["field"] = _field;
		result["value"] = _value;
		return result;
	};

	auto mkDirectMap = [&](std::vector<std::string> const& _path, Json const& _key, Json const& _value) {
		Json result = Json::object();
		result["kind"] = "storage_map_set";
		if (_path.size() == 1)
			result["field"] = _path.front();
		else
			result["path"] = jsonStringArray(_path);
		result["key"] = _key;
		result["value"] = _value;
		return result;
	};

	auto compoundValue = [&](Json const& _current, Json const& _rhsJson) {
		Json value = Json::object();
		switch (_op)
		{
		case Token::Assign:
			return _rhsJson;
		case Token::AssignAdd:
			value["kind"] = "u256_add";
			markUncheckedContext(value);
			// Compound assignment operates at the assigned expression's own
			// type (`a += b` is `a = a + b` at type(a)); the type checker
			// guarantees `b` is implicitly convertible to it.
			tagNarrowArithWidth(value, _lhs.annotation().type);
			break;
		case Token::AssignSub:
			value["kind"] = "u256_sub";
			markUncheckedContext(value);
			tagNarrowArithWidth(value, _lhs.annotation().type);
			break;
		case Token::AssignMul:
			value["kind"] = "u256_mul";
			markUncheckedContext(value);
			tagNarrowArithWidth(value, _lhs.annotation().type);
			break;
		case Token::AssignDiv:
			value["kind"] = "u256_div";
			break;
		case Token::AssignMod:
			value["kind"] = "u256_mod";
			break;
		case Token::AssignBitAnd:
			value["kind"] = "u256_bitand";
			break;
		case Token::AssignBitOr:
			value["kind"] = "u256_bitor";
			break;
		case Token::AssignBitXor:
			value["kind"] = "u256_bitxor";
			break;
		case Token::AssignShl:
			value["kind"] = "u256_shl";
			break;
		case Token::AssignSar:
			value["kind"] = "u256_shr";
			break;
		default:
			throw UnsupportedSolCore("Unsupported assignment operator in SolCore exporter.");
		}
		value["lhs"] = _current;
		value["rhs"] = _rhsJson;
		return value;
	};

	Json rhsJson = exportExpr(_rhs);

	if (auto const* identifier = dynamic_cast<Identifier const*>(&_lhs))
	{
		auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (!decl)
			throw UnsupportedSolCore("Assignment target without declaration.");
		if (decl->isStateVariable())
		{
			Json current = Json::object();
			current["kind"] = "storage_get";
			current["field"] = decl->name();
			return mkDirectStorage(decl->name(), compoundValue(current, rhsJson));
		}

		Json result = Json::object();
		result["kind"] = "assign";
		std::string varName = decl->name().empty() ? identifier->name() : decl->name();
		result["name"] = varName;
		if (_op != Token::Assign)
		{
			// Compound assignment to local: x += rhs → assign x = (x + rhs)
			Json current = Json::object();
			current["kind"] = "local";
			current["name"] = varName;
			result["value"] = compoundValue(current, rhsJson);
		}
		else
			result["value"] = rhsJson;
		return result;
	}

	if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_lhs))
	{
		// Check if base type is an array or fixed bytes (not a mapping)
		Type const* baseType = indexAccess->baseExpression().annotation().type;
		if (baseType && (baseType->category() == Type::Category::Array ||
		                 baseType->category() == Type::Category::FixedBytes))
		{
			if (!indexAccess->indexExpression())
				throw UnsupportedSolCore("Array index assignment without index.");
			// Compute the effective rhs (handling compound assignment)
			Json effectiveRhs = rhsJson;
			if (_op != Token::Assign)
			{
				Json current = Json::object();
				current["kind"] = "array_get";
				current["base"] = exportExpr(indexAccess->baseExpression());
				current["index"] = exportExpr(*indexAccess->indexExpression());
				effectiveRhs = compoundValue(current, rhsJson);
			}
			// Extract base_path as a string list for the OCaml parser
			if (auto const* baseIdent = dynamic_cast<Identifier const*>(&indexAccess->baseExpression()))
			{
				auto const* decl = dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
				// For local (memory) arrays, export as an assign of the whole array
				// using an internal_call to array_set, since the SolCore ArraySet node
				// only works with storage arrays.
				if (decl && !decl->isStateVariable())
				{
					// local_arr[idx] = val  →  assign local_arr = internal_call array_set(local_arr, idx, val)
					Json result = Json::object();
					result["kind"] = "assign";
					result["name"] = decl->name().empty() ? baseIdent->name() : decl->name();
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = "array_set_local";
					callExpr["args"] = Json::array();
					callExpr["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
					callExpr["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
					callExpr["args"].emplace_back(effectiveRhs);
					result["value"] = callExpr;
					return result;
				}
				// Storage array — use the standard array_set path
				Json basePath = Json::array();
				if (decl)
					basePath.emplace_back(decl->name());
				else
					basePath.emplace_back(baseIdent->name());
				Json result = Json::object();
				result["kind"] = "array_set";
				result["base_path"] = basePath;
				result["index"] = exportExpr(*indexAccess->indexExpression());
				result["value"] = effectiveRhs;
				return result;
			}
			else
			{
				// Complex base expression (e.g. result[expr] = val)
				// Export as an expr statement with internal_call array_set_local
				Json result = Json::object();
				result["kind"] = "expr";
				Json callExpr = Json::object();
				callExpr["kind"] = "internal_call";
				callExpr["function"] = "array_set_expr";
				callExpr["args"] = Json::array();
				callExpr["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
				callExpr["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
				callExpr["args"].emplace_back(effectiveRhs);
				result["value"] = callExpr;
				return result;
			}
		}

		// Fall back to mapping assignment
		try
		{
			auto [path, key] = exportStorageMapLValue(_lhs);
			Json current = Json::object();
			current["kind"] = "storage_map_get";
			if (path.size() == 1)
				current["field"] = path.front();
			else
				current["path"] = jsonStringArray(path);
			current["key"] = key;
			return mkDirectMap(path, key, compoundValue(current, rhsJson));
		}
		catch (...)
		{
			// Nested mapping assignment:
			// outer[innerKey][leafKey] = value
			// Export this as a storage_map_set on the outer mapping whose value is an
			// inner mapping update expression. This preserves the storage effect so the
			// downstream generator can thread state and compute accurate touched fields.
			try
			{
				auto [outerPath, outerKey] = exportStorageMapLValue(indexAccess->baseExpression());
				Json innerBase = exportExpr(indexAccess->baseExpression());
				Json leafIndex = exportExpr(*indexAccess->indexExpression());
				Json effectiveRhs = rhsJson;
				if (_op != Token::Assign)
				{
					Json current = Json::object();
					current["kind"] = "array_get";
					current["base"] = innerBase;
					current["index"] = leafIndex;
					effectiveRhs = compoundValue(current, rhsJson);
				}
				Json nestedUpdate = Json::object();
				nestedUpdate["kind"] = "internal_call";
				nestedUpdate["function"] = "array_set_expr";
				nestedUpdate["args"] = Json::array();
				nestedUpdate["args"].emplace_back(innerBase);
				nestedUpdate["args"].emplace_back(leafIndex);
				nestedUpdate["args"].emplace_back(effectiveRhs);
				return mkDirectMap(outerPath, outerKey, nestedUpdate);
			}
			catch (...)
			{
				// If nested mapping export also fails, fall back to generic array_set_expr
				Json result = Json::object();
				result["kind"] = "expr";
				Json callExpr = Json::object();
				callExpr["kind"] = "internal_call";
				callExpr["function"] = "array_set_expr";
				callExpr["args"] = Json::array();
				callExpr["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
				if (indexAccess->indexExpression())
					callExpr["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
				callExpr["args"].emplace_back(rhsJson);
				result["value"] = callExpr;
				return result;
			}
		}
	}

	// Member access LHS: structVar.field = value
	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_lhs))
	{
		// --- Namespaced storage: $.field = value → storage_set with prefixed field ---
		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			auto nsIt = namespacedStorageAliases.find(baseIdent->name());
			if (nsIt != namespacedStorageAliases.end())
			{
				std::string prefixedField = nsIt->second + memberAccess->memberName();
				Json current = Json::object();
				current["kind"] = "storage_get";
				current["field"] = prefixedField;
				return mkDirectStorage(prefixedField, compoundValue(current, rhsJson));
			}
		}
		// --- End namespaced storage member LHS ---

		auto exportStructUpdate = [&](Expression const& baseExpr) {
			// Compute the effective RHS (handling compound assignment)
			Json effectiveRhs = rhsJson;
			if (_op != Token::Assign)
			{
				Json current = Json::object();
				current["kind"] = "field";
				current["base"] = exportExpr(baseExpr);
				current["field"] = memberAccess->memberName();
				effectiveRhs = compoundValue(current, rhsJson);
			}

			// Export as: assign base = struct_update(base, field, value)
			Json update = Json::object();
			update["kind"] = "struct_update";
			update["base"] = exportExpr(baseExpr);
			update["field"] = memberAccess->memberName();
			update["value"] = effectiveRhs;
			return update;
		};

		// Check if the base is an identifier (local or state variable)
		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			auto const* decl = dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
			if (decl && !decl->isStateVariable())
			{
				// Local struct member write: $.field = value
				// Export as: assign $ = struct_update($, field, value)
				Json result = Json::object();
				result["kind"] = "assign";
				result["name"] = decl->name().empty() ? baseIdent->name() : decl->name();
				result["value"] = exportStructUpdate(memberAccess->expression());
				return result;
			}
			if (decl && decl->isStateVariable())
			{
				// State variable struct member write: stateVar.field = value
				// Export as: storage_set stateVar = struct_update(stateVar, field, value)
				Json result = Json::object();
				result["kind"] = "storage_set";
				result["field"] = decl->name();
				result["value"] = exportStructUpdate(memberAccess->expression());
				return result;
			}
		}
		// Generic member access assignment: export as expr with struct_update
		Json result = Json::object();
		result["kind"] = "expr";
		result["value"] = exportStructUpdate(memberAccess->expression());
		return result;
	}

	// Tuple LHS: (a, b) = (1, 2) — export as a block of individual assignments
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_lhs))
	{
		Json result = Json::object();
		result["kind"] = "expr";
		Json callExpr = Json::object();
		callExpr["kind"] = "internal_call";
		callExpr["function"] = "tuple_assign";
		callExpr["args"] = Json::array();
		// [SolCore audit finding #9] No catch-and-substitute-0 here: an
		// unlowerable tuple-assignment-target component must propagate to
		// exportBody's catch-all rather than silently assigning through an
		// untraceable zero target.
		for (auto const& component: tuple->components())
		{
			if (component)
				callExpr["args"].emplace_back(exportExpr(*component));
			else
				callExpr["args"].emplace_back(Json());
		}
		callExpr["args"].emplace_back(rhsJson);
		result["value"] = callExpr;
		return result;
	}

	// Fallback: export as generic expr instead of throwing
	{
		Json result = Json::object();
		result["kind"] = "expr";
		Json callExpr = Json::object();
		callExpr["kind"] = "internal_call";
		callExpr["function"] = "generic_assign";
		callExpr["args"] = Json::array();
		// [SolCore audit finding #9] No catch-and-substitute-0 here: an
		// unlowerable assignment target must propagate to exportBody's
		// catch-all rather than silently assigning through an untraceable
		// zero target.
		callExpr["args"].emplace_back(exportExpr(_lhs));
		callExpr["args"].emplace_back(rhsJson);
		result["value"] = callExpr;
		return result;
	}
}

Json exportDirectAssignment(Expression const& _lhs, Json const& _value)
{
	auto mkDirectStorage = [&](std::string const& _field, Json const& _storedValue) {
		Json result = Json::object();
		result["kind"] = "storage_set";
		result["field"] = _field;
		result["value"] = _storedValue;
		return result;
	};

	auto mkDirectMap = [&](std::vector<std::string> const& _path, Json const& _key, Json const& _storedValue) {
		Json result = Json::object();
		result["kind"] = "storage_map_set";
		if (_path.size() == 1)
			result["field"] = _path.front();
		else
			result["path"] = jsonStringArray(_path);
		result["key"] = _key;
		result["value"] = _storedValue;
		return result;
	};

	if (auto const* identifier = dynamic_cast<Identifier const*>(&_lhs))
	{
		auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (!decl)
			throw UnsupportedSolCore("Assignment target without declaration.");
		if (decl->isStateVariable())
			return mkDirectStorage(decl->name(), _value);

		Json result = Json::object();
		result["kind"] = "assign";
		result["name"] = decl->name().empty() ? identifier->name() : decl->name();
		result["value"] = _value;
		return result;
	}

	if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_lhs))
	{
		Type const* baseType = indexAccess->baseExpression().annotation().type;
		if (baseType && (baseType->category() == Type::Category::Array ||
		                 baseType->category() == Type::Category::FixedBytes))
		{
			if (!indexAccess->indexExpression())
				throw UnsupportedSolCore("Array index assignment without index.");
			if (auto const* baseIdent = dynamic_cast<Identifier const*>(&indexAccess->baseExpression()))
			{
				auto const* decl = dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
				if (decl && !decl->isStateVariable())
				{
					Json result = Json::object();
					result["kind"] = "assign";
					result["name"] = decl->name().empty() ? baseIdent->name() : decl->name();
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = "array_set_local";
					callExpr["args"] = Json::array();
					callExpr["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
					callExpr["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
					callExpr["args"].emplace_back(_value);
					result["value"] = callExpr;
					return result;
				}
				Json basePath = Json::array();
				if (decl)
					basePath.emplace_back(decl->name());
				else
					basePath.emplace_back(baseIdent->name());
				Json result = Json::object();
				result["kind"] = "array_set";
				result["base_path"] = basePath;
				result["index"] = exportExpr(*indexAccess->indexExpression());
				result["value"] = _value;
				return result;
			}
			Json result = Json::object();
			result["kind"] = "expr";
			Json callExpr = Json::object();
			callExpr["kind"] = "internal_call";
			callExpr["function"] = "array_set_expr";
			callExpr["args"] = Json::array();
			callExpr["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
			callExpr["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
			callExpr["args"].emplace_back(_value);
			result["value"] = callExpr;
			return result;
		}

		try
		{
			auto [path, key] = exportStorageMapLValue(_lhs);
			return mkDirectMap(path, key, _value);
		}
		catch (...)
		{
			try
			{
				auto [outerPath, outerKey] = exportStorageMapLValue(indexAccess->baseExpression());
				Json nestedUpdate = Json::object();
				nestedUpdate["kind"] = "internal_call";
				nestedUpdate["function"] = "array_set_expr";
				nestedUpdate["args"] = Json::array();
				nestedUpdate["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
				nestedUpdate["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
				nestedUpdate["args"].emplace_back(_value);
				return mkDirectMap(outerPath, outerKey, nestedUpdate);
			}
			catch (...)
			{
				Json result = Json::object();
				result["kind"] = "expr";
				Json callExpr = Json::object();
				callExpr["kind"] = "internal_call";
				callExpr["function"] = "array_set_expr";
				callExpr["args"] = Json::array();
				callExpr["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
				if (indexAccess->indexExpression())
					callExpr["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
				callExpr["args"].emplace_back(_value);
				result["value"] = callExpr;
				return result;
			}
		}
	}

	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_lhs))
	{
		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			auto nsIt = namespacedStorageAliases.find(baseIdent->name());
			if (nsIt != namespacedStorageAliases.end())
			{
				std::string prefixedField = nsIt->second + memberAccess->memberName();
				return mkDirectStorage(prefixedField, _value);
			}
		}

		auto exportStructUpdate = [&](Expression const& baseExpr) {
			Json update = Json::object();
			update["kind"] = "struct_update";
			update["base"] = exportExpr(baseExpr);
			update["field"] = memberAccess->memberName();
			update["value"] = _value;
			return update;
		};

		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			auto const* decl = dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
			if (decl && !decl->isStateVariable())
			{
				Json result = Json::object();
				result["kind"] = "assign";
				result["name"] = decl->name().empty() ? baseIdent->name() : decl->name();
				result["value"] = exportStructUpdate(memberAccess->expression());
				return result;
			}
			if (decl && decl->isStateVariable())
			{
				Json result = Json::object();
				result["kind"] = "storage_set";
				result["field"] = decl->name();
				result["value"] = exportStructUpdate(memberAccess->expression());
				return result;
			}
		}
		Json result = Json::object();
		result["kind"] = "expr";
		result["value"] = exportStructUpdate(memberAccess->expression());
		return result;
	}

	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_lhs))
	{
		Json result = Json::object();
		result["kind"] = "expr";
		Json callExpr = Json::object();
		callExpr["kind"] = "internal_call";
		callExpr["function"] = "tuple_assign";
		callExpr["args"] = Json::array();
		// [SolCore audit finding #9] No catch-and-substitute-0 here: an
		// unlowerable tuple-assignment-target component must propagate to
		// exportBody's catch-all rather than silently assigning through an
		// untraceable zero target.
		for (auto const& component: tuple->components())
		{
			if (component)
				callExpr["args"].emplace_back(exportExpr(*component));
			else
				callExpr["args"].emplace_back(Json());
		}
		callExpr["args"].emplace_back(_value);
		result["value"] = callExpr;
		return result;
	}

	Json result = Json::object();
	result["kind"] = "expr";
	Json callExpr = Json::object();
	callExpr["kind"] = "internal_call";
	callExpr["function"] = "generic_assign";
	callExpr["args"] = Json::array();
	// [SolCore audit finding #9] No catch-and-substitute-0 here: an
	// unlowerable assignment target must propagate to exportBody's
	// catch-all rather than silently assigning through an untraceable zero
	// target.
	callExpr["args"].emplace_back(exportExpr(_lhs));
	callExpr["args"].emplace_back(_value);
	result["value"] = callExpr;
	return result;
}

Json mutationValue(Json const& _current, Token _op, Type const* _targetType)
{
	Json result = Json::object();
	switch (_op)
	{
	case Token::Inc:
		result["kind"] = "u256_add";
		markUncheckedContext(result);
		tagNarrowArithWidth(result, _targetType);
		break;
	case Token::Dec:
		result["kind"] = "u256_sub";
		markUncheckedContext(result);
		tagNarrowArithWidth(result, _targetType);
		break;
	default:
		throw UnsupportedSolCore("Expected ++ or -- unary mutation.");
	}
	result["lhs"] = _current;
	result["rhs"] = u256Literal("1");
	return result;
}

Json exportUnaryMutation(Expression const& _target, Token _op)
{
	return exportDirectAssignment(_target, mutationValue(exportExpr(_target), _op, _target.annotation().type));
}

Json exportUnaryMutationReturn(UnaryOperation const& _unary)
{
	Expression const& target = _unary.subExpression();
	std::string tempName = "__solcore_tmp_" + std::to_string(_unary.id());

	Json block = Json::object();
	block["kind"] = "block";
	block["statements"] = Json::array();

	if (_unary.isPrefixOperation())
	{
		Json letStmt = Json::object();
		letStmt["kind"] = "let";
		letStmt["name"] = tempName;
		letStmt["type"] = "u256";
		letStmt["value"] = mutationValue(exportExpr(target), _unary.getOperator(), target.annotation().type);
		block["statements"].emplace_back(std::move(letStmt));
		block["statements"].emplace_back(exportDirectAssignment(target, localExpr(tempName)));
	}
	else
	{
		Json letStmt = Json::object();
		letStmt["kind"] = "let";
		letStmt["name"] = tempName;
		letStmt["type"] = "u256";
		letStmt["value"] = exportExpr(target);
		block["statements"].emplace_back(std::move(letStmt));
		block["statements"].emplace_back(exportDirectAssignment(target, mutationValue(localExpr(tempName), _unary.getOperator(), target.annotation().type)));
	}

	Json ret = Json::object();
	ret["kind"] = "return";
	ret["value"] = localExpr(tempName);
	block["statements"].emplace_back(std::move(ret));
	return block;
}

// --- Yul AST export functions ---

Json exportYulExpr(yul::Expression const& _expr, yul::Dialect const& _dialect);
Json exportYulStmt(yul::Statement const& _stmt, yul::Dialect const& _dialect);
Json exportYulBlock(yul::Block const& _block, yul::Dialect const& _dialect);

Json exportYulExpr(yul::Expression const& _expr, yul::Dialect const& _dialect)
{
	return std::visit(util::GenericVisitor{
		[&](yul::Literal const& _literal) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_literal";
			result["value"] = yul::formatLiteral(_literal);
			result["type"] = "u256";
			return result;
		},
		[&](yul::Identifier const& _identifier) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_local";
			result["name"] = _identifier.name.str();
			return result;
		},
		[&](yul::FunctionCall const& _call) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_call";
			result["function"] = std::string(yul::resolveFunctionName(_call.functionName, _dialect));
			result["args"] = Json::array();
			for (auto const& arg: _call.arguments)
				result["args"].emplace_back(exportYulExpr(arg, _dialect));
			return result;
		}
	}, _expr);
}

Json exportYulBlock(yul::Block const& _block, yul::Dialect const& _dialect)
{
	Json result = Json::object();
	result["kind"] = "yul_block";
	result["statements"] = Json::array();
	for (auto const& stmt: _block.statements)
		result["statements"].emplace_back(exportYulStmt(stmt, _dialect));
	return result;
}

Json exportYulStmt(yul::Statement const& _stmt, yul::Dialect const& _dialect)
{
	return std::visit(util::GenericVisitor{
		[&](yul::ExpressionStatement const& _exprStmt) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_expr";
			result["value"] = exportYulExpr(_exprStmt.expression, _dialect);
			return result;
		},
		[&](yul::Assignment const& _assignment) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_assign";
			result["names"] = Json::array();
			for (auto const& var: _assignment.variableNames)
				result["names"].emplace_back(var.name.str());
			if (_assignment.value)
				result["value"] = exportYulExpr(*_assignment.value, _dialect);
			else
				result["value"] = Json();
			return result;
		},
		[&](yul::VariableDeclaration const& _varDecl) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_let";
			result["names"] = Json::array();
			for (auto const& var: _varDecl.variables)
				result["names"].emplace_back(var.name.str());
			if (_varDecl.value)
				result["value"] = exportYulExpr(*_varDecl.value, _dialect);
			else
				result["value"] = Json();
			return result;
		},
		[&](yul::FunctionDefinition const& _funDef) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_fundef";
			result["name"] = _funDef.name.str();
			result["params"] = Json::array();
			for (auto const& param: _funDef.parameters)
				result["params"].emplace_back(param.name.str());
			result["returns"] = Json::array();
			for (auto const& ret: _funDef.returnVariables)
				result["returns"].emplace_back(ret.name.str());
			result["body"] = exportYulBlock(_funDef.body, _dialect);
			return result;
		},
		[&](yul::If const& _if) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_if";
			result["cond"] = exportYulExpr(*_if.condition, _dialect);
			result["body"] = exportYulBlock(_if.body, _dialect);
			return result;
		},
		[&](yul::Switch const& _switch) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_switch";
			result["expr"] = exportYulExpr(*_switch.expression, _dialect);
			result["cases"] = Json::array();
			for (auto const& c: _switch.cases)
			{
				Json caseJson = Json::object();
				if (c.value)
				{
					Json litJson = Json::object();
					litJson["kind"] = "yul_literal";
					litJson["value"] = yul::formatLiteral(*c.value);
					litJson["type"] = "u256";
					caseJson["value"] = litJson;
				}
				else
					caseJson["value"] = Json();
				caseJson["body"] = exportYulBlock(c.body, _dialect);
				result["cases"].emplace_back(caseJson);
			}
			return result;
		},
		[&](yul::ForLoop const& _for) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_for";
			result["pre"] = exportYulBlock(_for.pre, _dialect);
			result["cond"] = exportYulExpr(*_for.condition, _dialect);
			result["post"] = exportYulBlock(_for.post, _dialect);
			result["body"] = exportYulBlock(_for.body, _dialect);
			return result;
		},
		[&](yul::Break const&) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_break";
			return result;
		},
		[&](yul::Continue const&) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_continue";
			return result;
		},
		[&](yul::Leave const&) -> Json {
			Json result = Json::object();
			result["kind"] = "yul_leave";
			return result;
		},
		[&](yul::Block const& _block) -> Json {
			return exportYulBlock(_block, _dialect);
		}
	}, _stmt);
}

// --- End Yul AST export functions ---

Json exportStmt(Statement const& _stmt)
{
	if (auto const* block = dynamic_cast<Block const*>(&_stmt))
	{
		UncheckedBlockGuard uncheckedGuard(block->unchecked());
		Json result = Json::object();
		result["kind"] = "block";
		if (block->unchecked())
			result["unchecked"] = true;
		result["statements"] = Json::array();
		for (auto const& statement: block->statements())
			result["statements"].emplace_back(exportStmt(*statement));
		return result;
	}

	if (auto const* ifStmt = dynamic_cast<IfStatement const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "if";
		result["cond"] = exportExpr(ifStmt->condition());
		result["then"] = exportStmt(ifStmt->trueStatement());
		if (ifStmt->falseStatement())
			result["else"] = exportStmt(*ifStmt->falseStatement());
		else
		{
			Json emptyElse = Json::object();
			emptyElse["kind"] = "block";
			emptyElse["statements"] = Json::array();
			result["else"] = emptyElse;
		}
		return result;
	}

	if (auto const* returnStmt = dynamic_cast<Return const*>(&_stmt))
	{
		if (returnStmt->expression())
			if (auto const* unary = dynamic_cast<UnaryOperation const*>(returnStmt->expression()))
				if (unary->getOperator() == Token::Inc || unary->getOperator() == Token::Dec)
					return exportUnaryMutationReturn(*unary);

		Json result = Json::object();
		result["kind"] = "return";
		if (returnStmt->expression())
			result["value"] = exportExpr(*returnStmt->expression());
		else
		{
			Json unit = Json::object();
			unit["kind"] = "unit";
			result["value"] = unit;
		}
		return result;
	}

	if (auto const* varDecl = dynamic_cast<VariableDeclarationStatement const*>(&_stmt))
	{
		// --- Namespaced storage: detect `TokenStorage storage $ = _getTokenStorage();`
		// and register `$` as a namespaced storage alias instead of emitting a let statement.
		if (varDecl->declarations().size() == 1 && varDecl->declarations().front() && varDecl->initialValue())
		{
			auto const& decl = *varDecl->declarations().front();
			if (decl.referenceLocation() == VariableDeclaration::Location::Storage)
			{
				// Check if the initializer is a call to a namespaced storage getter
				if (auto const* call = dynamic_cast<FunctionCall const*>(varDecl->initialValue()))
				{
					if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
					{
						if (auto const* prefix = namespacedGetterPrefix(*callee))
						{
							// Register this local variable as a namespaced storage alias
							namespacedStorageAliases[decl.name()] = *prefix;
							// Emit a no-op block instead of a let statement
							Json result = Json::object();
							result["kind"] = "block";
							result["statements"] = Json::array();
							return result;
						}
					}
				}
			}
		}
		// --- End namespaced storage var decl ---

		// Single variable declaration with initializer (common case)
		if (varDecl->declarations().size() == 1 && varDecl->declarations().front() && varDecl->initialValue())
		{
			Json result = Json::object();
			result["kind"] = "let";
			result["name"] = varDecl->declarations().front()->name();
			try
			{
				result["type"] = exportTypeName(varDecl->declarations().front()->typeName());
			}
			catch (...)
			{
				result["type"] = Json("u256");
			}
			result["value"] = exportExpr(*varDecl->initialValue());
			return result;
		}
		// Single variable declaration without initializer: uint256 x;
		if (varDecl->declarations().size() == 1 && varDecl->declarations().front() && !varDecl->initialValue())
		{
			Json result = Json::object();
			result["kind"] = "let";
			result["name"] = varDecl->declarations().front()->name();
			try
			{
				result["type"] = exportTypeName(varDecl->declarations().front()->typeName());
				result["value"] = defaultValueForTypeName(varDecl->declarations().front()->typeName());
			}
			catch (...)
			{
				result["type"] = Json("u256");
				Json zero = Json::object();
				zero["kind"] = "u256";
				zero["value"] = "0";
				result["value"] = zero;
			}
			return result;
		}
		// Multiple variable declarations (tuple destructuring): (uint a, uint b) = f()
		{
			Json result = Json::object();
			result["kind"] = "block";
			result["statements"] = Json::array();

			// First, evaluate the initializer expression (if any) to a temporary
			Json initValue;
			if (varDecl->initialValue())
				initValue = exportExpr(*varDecl->initialValue());

			for (size_t i = 0; i < varDecl->declarations().size(); ++i)
			{
				auto const& decl = varDecl->declarations()[i];
				if (!decl)
					continue;
				Json letStmt = Json::object();
				letStmt["kind"] = "let";
				letStmt["name"] = decl->name();
				try
				{
					letStmt["type"] = exportTypeName(decl->typeName());
				}
				catch (...)
				{
					letStmt["type"] = Json("u256");
				}
				if (varDecl->initialValue())
				{
					// For tuple returns, extract the i-th element
					Json extractExpr = Json::object();
					extractExpr["kind"] = "internal_call";
					extractExpr["function"] = "tuple_get";
					extractExpr["args"] = Json::array();
					extractExpr["args"].emplace_back(initValue);
					Json idx = Json::object();
					idx["kind"] = "u256";
					idx["value"] = std::to_string(i);
					extractExpr["args"].emplace_back(idx);
					letStmt["value"] = extractExpr;
				}
				else
				{
					try
					{
						letStmt["value"] = defaultValueForTypeName(decl->typeName());
					}
					catch (...)
					{
						Json zero = Json::object();
						zero["kind"] = "u256";
						zero["value"] = "0";
						letStmt["value"] = zero;
					}
				}
				result["statements"].emplace_back(letStmt);
			}
			return result;
		}
	}

	if (auto const* expressionStmt = dynamic_cast<ExpressionStatement const*>(&_stmt))
	{
		Expression const& expr = expressionStmt->expression();

		if (auto const* assignment = dynamic_cast<Assignment const*>(&expr))
			return exportAssignment(assignment->leftHandSide(), assignment->assignmentOperator(), assignment->rightHandSide());

		if (auto const* unary = dynamic_cast<UnaryOperation const*>(&expr))
			if (unary->getOperator() == Token::Inc || unary->getOperator() == Token::Dec)
				return exportUnaryMutation(unary->subExpression(), unary->getOperator());

		if (auto const* call = dynamic_cast<FunctionCall const*>(&expr))
		{
			if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
			{
				if (callee->name() == "require")
				{
					if (call->arguments().empty())
						throw UnsupportedSolCore("require() without arguments is unsupported.");
					Json result = Json::object();
					result["kind"] = "require";
					result["cond"] = exportExpr(*call->arguments().front());
					Json payload = exportRevertPayload(*call);
					for (auto const& [key, value]: payload.items())
						result[key] = value;
					return result;
				}

				// assert(cond) as statement
				if (callee->name() == "assert")
				{
					if (call->arguments().empty())
						throw UnsupportedSolCore("assert() without arguments is unsupported.");
					Json result = Json::object();
					result["kind"] = "assert";
					result["cond"] = exportExpr(*call->arguments().front());
					return result;
				}

				// revert("message") as statement (without RevertStatement syntax)
				if (callee->name() == "revert")
				{
					Json result = Json::object();
					result["kind"] = "revert";
					Json payload = exportRevertPayload(*call);
					for (auto const& [key, value]: payload.items())
						result[key] = value;
					return result;
				}

				// selfdestruct(address) as statement
				if (callee->name() == "selfdestruct")
				{
					if (call->arguments().empty())
						throw UnsupportedSolCore("selfdestruct() without arguments is unsupported.");
					Json result = Json::object();
					result["kind"] = "selfdestruct";
					result["target"] = exportExpr(*call->arguments().front());
					return result;
				}

				// Internal function call as statement. Plain-identifier calls
				// are VIRTUAL dispatch — name the slot winner, not the
				// lexically referenced declaration (see virtualCallTargetName).
				auto const* funcDef = dynamic_cast<FunctionDefinition const*>(callee->annotation().referencedDeclaration);
				if (funcDef)
				{
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = virtualCallTargetName(*funcDef);
					callExpr["args"] = Json::array();
					for (auto const& arg: call->arguments())
						callExpr["args"].emplace_back(exportExpr(*arg));
					result["value"] = callExpr;
					return result;
				}
			}

			// Array push: arr.push(value), Array pop: arr.pop()
			if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
			{
				// Statically-bound super/base-qualified internal call as a
				// statement — same handling as the expression path above (and
				// it MUST run before the library/type-qualified branch below,
				// which would otherwise collapse it onto the virtual-slot
				// name and emit a self-forward).
				if (auto resolved = resolveStaticBaseCallTarget(*call, *memberAccess, activeExportContract))
				{
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = flattenedStaticBaseCallName(*resolved);
					callExpr["args"] = Json::array();
					for (auto const& arg: call->arguments())
						callExpr["args"].emplace_back(exportExpr(*arg));
					result["value"] = callExpr;
					return result;
				}

				if (memberAccess->memberName() == "push" || memberAccess->memberName() == "pop")
				{
					Type const* baseType = memberAccess->expression().annotation().type;
					if (baseType && baseType->category() == Type::Category::Array)
					{
						// Extract base_path as a string list for the OCaml parser
						Json basePath = Json::array();
						if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
						{
							auto const* decl = dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
							if (decl && decl->isStateVariable())
								basePath.emplace_back(decl->name());
							else if (decl)
								throw UnsupportedSolCore("Array push/pop on local storage reference is unsupported.");
							else
								basePath.emplace_back(baseIdent->name());
						}
						else
							throw UnsupportedSolCore("Array push/pop statement with complex base expression is unsupported.");

						if (memberAccess->memberName() == "push")
						{
							Json result = Json::object();
							result["kind"] = "array_push";
							result["base_path"] = basePath;
							if (!call->arguments().empty())
								result["value"] = exportExpr(*call->arguments().front());
							else
							{
								// push() without arguments pushes a default value
								Json defaultVal = Json::object();
								defaultVal["kind"] = "u256";
								defaultVal["value"] = "0";
								result["value"] = defaultVal;
							}
							return result;
						}
						else // pop
						{
							Json result = Json::object();
							result["kind"] = "array_pop";
							// parse_path ~field_name:"base" checks "path" first, then "base" as string
							result["path"] = basePath;
							return result;
						}
					}
				}

				// using-for library extension call as statement.
				if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
				{
					Type const* baseType = memberAccess->expression().annotation().type;
					if (
						baseType &&
						baseType->category() != Type::Category::TypeType &&
						baseType->category() != Type::Category::Module &&
						funcDef->visibility() <= Visibility::Internal
					)
					{
						Json result = Json::object();
						result["kind"] = "expr";
						Json callExpr = Json::object();
						callExpr["kind"] = "internal_call";
						callExpr["function"] = exportedFunctionName(*funcDef);
						callExpr["args"] = Json::array();
						callExpr["args"].emplace_back(exportExpr(memberAccess->expression()));
						for (auto const& arg: call->arguments())
							callExpr["args"].emplace_back(exportExpr(*arg));
						result["value"] = callExpr;
						return result;
					}
				}

				// External contract call as statement: contract.method(args)
				Type const* baseType = memberAccess->expression().annotation().type;
				if (baseType && baseType->category() == Type::Category::Contract)
				{
					if (auto externalCall = exportExternalContractCall(*call, *memberAccess, true))
						return *externalCall;
				}

				// Library-qualified or type-qualified function call as statement: L.f(args)
				if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
				{
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = exportedFunctionName(*funcDef);
					callExpr["args"] = Json::array();
					for (auto const& arg: call->arguments())
						callExpr["args"].emplace_back(exportExpr(*arg));
					result["value"] = callExpr;
					return result;
				}
			}

			// Low-level call as statement: address.call{value: ...}(data) or delegatecall
			if (auto const* options = dynamic_cast<FunctionCallOptions const*>(&call->expression()))
			{
				if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&options->expression()))
				{
					if (auto externalCall = exportExternalContractCall(*call, *memberAccess, true))
						return *externalCall;

					if (memberAccess->memberName() == "call" || memberAccess->memberName() == "delegatecall")
					{
						Json result = Json::object();
						result["kind"] = "expr";
						Json callExpr = Json::object();
						callExpr["kind"] = "low_level_call";
						callExpr["callKind"] = lowLevelCallKindString(memberAccess->memberName());
						callExpr["target"] = exportExpr(memberAccess->expression());

						// Extract value from options
						Json valueExpr = Json::object();
						valueExpr["kind"] = "u256";
						valueExpr["value"] = "0";
						for (size_t i = 0; i < options->names().size(); ++i)
						{
							if (*options->names()[i] == "value")
							{
								valueExpr = exportExpr(*options->options()[i]);
								break;
							}
						}
						callExpr["value"] = valueExpr;

						// Data is the first argument to the call
						if (!call->arguments().empty())
							callExpr["data"] = exportExpr(*call->arguments().front());
						else
						{
							Json emptyData = Json::object();
							emptyData["kind"] = "string";
							emptyData["value"] = "";
							callExpr["data"] = emptyData;
						}
						result["value"] = callExpr;
						return result;
					}
				}
			}

			// Low-level call as statement without options: address.call(data) or address.delegatecall(data)
			if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
			{
				if (memberAccess->memberName() == "call" || memberAccess->memberName() == "delegatecall")
				{
					Type const* baseType = memberAccess->expression().annotation().type;
					if (baseType && baseType->category() == Type::Category::Address)
					{
						Json result = Json::object();
						result["kind"] = "expr";
						Json callExpr = Json::object();
						callExpr["kind"] = "low_level_call";
						callExpr["callKind"] = lowLevelCallKindString(memberAccess->memberName());
						callExpr["target"] = exportExpr(memberAccess->expression());

						Json valueExpr = Json::object();
						valueExpr["kind"] = "u256";
						valueExpr["value"] = "0";
						callExpr["value"] = valueExpr;

						if (!call->arguments().empty())
							callExpr["data"] = exportExpr(*call->arguments().front());
						else
						{
							Json emptyData = Json::object();
							emptyData["kind"] = "string";
							emptyData["value"] = "";
							callExpr["data"] = emptyData;
						}
						result["value"] = callExpr;
						return result;
					}
				}
			}
		}

		Json result = Json::object();
		result["kind"] = "expr";
		result["value"] = exportExpr(expr);
		return result;
	}

	if (auto const* emitStmt = dynamic_cast<EmitStatement const*>(&_stmt))
	{
		FunctionCall const& call = emitStmt->eventCall();
		Json result = Json::object();
		result["kind"] = "emit";

		// Extract event name from the call expression
		if (auto const* callee = dynamic_cast<Identifier const*>(&call.expression()))
		{
			result["event"] = callee->name();
		}
		else if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call.expression()))
		{
			result["event"] = memberAccess->memberName();
		}
		else
			throw UnsupportedSolCore("Unsupported emit expression in SolCore exporter.");

		result["args"] = Json::array();
		for (auto const& arg: call.arguments())
			result["args"].emplace_back(exportExpr(*arg));
		return result;
	}

	if (auto const* revertStmt = dynamic_cast<RevertStatement const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "revert";
		Json payload = exportRevertPayload(revertStmt->errorCall());
		for (auto const& [key, value]: payload.items())
			result[key] = value;
		return result;
	}

	if (auto const* asmStmt = dynamic_cast<InlineAssembly const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "inline_assembly";
		result["body"] = exportYulBlock(asmStmt->operations().root(), asmStmt->dialect());
		return result;
	}

	if (dynamic_cast<PlaceholderStatement const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "placeholder";
		return result;
	}

	if (auto const* whileStmt = dynamic_cast<WhileStatement const*>(&_stmt))
	{
		Json result = Json::object();
		if (whileStmt->isDoWhile())
			result["kind"] = "do_while";
		else
			result["kind"] = "while";
		result["cond"] = exportExpr(whileStmt->condition());
		result["body"] = exportStmt(whileStmt->body());
		return result;
	}

	if (auto const* forStmt = dynamic_cast<ForStatement const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "for";
		if (forStmt->initializationExpression())
			result["init"] = exportStmt(*forStmt->initializationExpression());
		else
			result["init"] = Json();
		if (forStmt->condition())
			result["cond"] = exportExpr(*forStmt->condition());
		else
			result["cond"] = Json();
		if (forStmt->loopExpression())
		{
			// loopExpression() returns an ExpressionStatement; export its inner expression as a stmt
			result["post"] = exportStmt(*forStmt->loopExpression());
		}
		else
			result["post"] = Json();
		result["body"] = exportStmt(forStmt->body());
		return result;
	}

	if (dynamic_cast<Break const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "break";
		return result;
	}

	if (dynamic_cast<Continue const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "continue";
		return result;
	}

	if (auto const* tryStmt = dynamic_cast<TryStatement const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "try_catch";
		result["call"] = exportExpr(tryStmt->externalCall());

		Json clauses = Json::array();
		for (auto const& clause : tryStmt->clauses())
		{
			Json c = Json::object();
			c["error_name"] = clause->errorName().empty() ? Json() : Json(clause->errorName());

			if (clause->parameters())
			{
				c["params"] = Json::array();
				for (auto const& param : clause->parameters()->parameters())
					c["params"].emplace_back(exportParam(*param));
			}

			c["body"] = exportStmt(clause->block());
			clauses.emplace_back(c);
		}
		result["clauses"] = clauses;
		return result;
	}

	// Fallback: export as a no-op block rather than throwing.
	// This allows partial export of functions that contain an unsupported
	// statement type, rather than failing the entire function.
	{
		Json result = Json::object();
		result["kind"] = "block";
		result["statements"] = Json::array();
		result["_unsupported_stmt"] = true;
		return result;
	}
}

// ===========================================================================
// AST write-set oracle (Layer 2 of the SOL-PLAN-FIDELITY cross-check).
//
// This is a SECOND, INDEPENDENT pass over the Solidity AST that estimates
// which state variables a function's execution can write, computed WITHOUT
// using exportStmt/exportExpr/exportBody at all. The entire point is that it
// must not be able to inherit exportStmt/exportExpr's bugs (in particular
// the catch-all body-export failure exportBody() now guards against above):
// if the JSON body export silently produced a wrong (too-weak) model, this
// walk gives the OCaml frontend an independent second opinion to diff
// against. It is deliberately conservative: anything it cannot resolve
// statically sets `unknown`, it never guesses, and it must never throw.
// ===========================================================================

struct WriteOracleResult
{
	std::set<std::string> writes;
	bool unknown = false;
};

// Resolve a function-call callee expression (Identifier or MemberAccess) to
// the FunctionDefinition that will actually execute, applying virtual
// dispatch against `_mostDerivedContract` when the call site requires
// virtual lookup. This generalizes the resolution idiom already used by
// resolveModifierDefinition() above (ModifierDefinition::resolveVirtual)
// to FunctionDefinition::resolveVirtual.
FunctionDefinition const* resolveWriteOracleCallTarget(
	Expression const& _callee,
	ContractDefinition const* _mostDerivedContract)
{
	Declaration const* referenced = nullptr;
	bool requiresVirtual = false;
	if (auto const* ident = dynamic_cast<Identifier const*>(&_callee))
	{
		referenced = ident->annotation().referencedDeclaration;
		requiresVirtual =
			ident->annotation().requiredLookup.set() &&
			*ident->annotation().requiredLookup == VirtualLookup::Virtual;
	}
	else if (auto const* member = dynamic_cast<MemberAccess const*>(&_callee))
	{
		referenced = member->annotation().referencedDeclaration;
		requiresVirtual =
			member->annotation().requiredLookup.set() &&
			*member->annotation().requiredLookup == VirtualLookup::Virtual;
	}
	else
		return nullptr;

	auto const* funcDef = dynamic_cast<FunctionDefinition const*>(referenced);
	if (!funcDef)
		return nullptr;
	if (requiresVirtual && _mostDerivedContract)
		return &funcDef->resolveVirtual(*_mostDerivedContract);
	return funcDef;
}

// Scan a Yul AST for sstore, independently of exportYulExpr/exportYulStmt
// above (same std::visit idiom, but this one only answers "does this touch
// PERSISTENT storage", it does not build any JSON). We do not try to
// resolve which slot(s) are touched — any occurrence is treated as a fully
// opaque write, matching the existing OCaml havoc-modeling policy for
// assembly blocks (SolCore.ml's `inline_asm_writes_persistent_storage`
// marks any `assembly { ... }` block containing `sstore` as a
// StorageHavocUpdate on the OCaml side; this oracle independently confirms
// the same conclusion from the raw AST rather than trusting that pass).
//
// `tstore` (EIP-1153 transient storage) is deliberately NOT matched here:
// transient storage is a sibling of the named-field `Storage` model this
// oracle protects (`ast_write_oracle` feeds ProofPlan's
// `model_fidelity_diag`, which only ever compares against
// `transitive_touched_fields`, itself a set of NAMED PERSISTENT fields — see
// SOLCORE_TLOAD_TSTORE_ARCHITECTURE_SPEC.md §4.5). A tstore-only assembly
// block genuinely cannot perturb persistent storage, so flagging it
// `unknown` here would falsely demote every function reachable from it
// (e.g. OpenZeppelin's `ReentrancyGuardTransient._nonReentrantBefore`) to
// `Unsupported.SOL-PLAN-FIDELITY-002` even though the OCaml/Lean side now
// proves persistent-storage preservation through it correctly. Keeping
// this oracle's `sstore`-only judgment in sync with the OCaml split is
// exactly why this comment (and the one on `WriteOracleCollector::visit
// (InlineAssembly const&)` below) call out the pairing explicitly.
bool yulExprMayStoreToStorage(yul::Expression const& _expr, yul::Dialect const& _dialect);
bool yulStmtMayStoreToStorage(yul::Statement const& _stmt, yul::Dialect const& _dialect);

bool yulBlockMayStoreToStorage(yul::Block const& _block, yul::Dialect const& _dialect)
{
	for (auto const& stmt: _block.statements)
		if (yulStmtMayStoreToStorage(stmt, _dialect))
			return true;
	return false;
}

bool yulExprMayStoreToStorage(yul::Expression const& _expr, yul::Dialect const& _dialect)
{
	return std::visit(util::GenericVisitor{
		[&](yul::Literal const&) -> bool { return false; },
		[&](yul::Identifier const&) -> bool { return false; },
		[&](yul::FunctionCall const& _call) -> bool {
			std::string name = std::string(yul::resolveFunctionName(_call.functionName, _dialect));
			// `tstore` writes EIP-1153 transient storage, not the named
			// PERSISTENT `Storage` fields this oracle exists to protect — see
			// the comment above `yulExprMayStoreToStorage`'s declaration for
			// why it is deliberately excluded here.
			if (name == "sstore")
				return true;
			for (auto const& arg: _call.arguments)
				if (yulExprMayStoreToStorage(arg, _dialect))
					return true;
			return false;
		}
	}, _expr);
}

bool yulStmtMayStoreToStorage(yul::Statement const& _stmt, yul::Dialect const& _dialect)
{
	return std::visit(util::GenericVisitor{
		[&](yul::ExpressionStatement const& _exprStmt) -> bool {
			return yulExprMayStoreToStorage(_exprStmt.expression, _dialect);
		},
		[&](yul::Assignment const& _assignment) -> bool {
			return _assignment.value && yulExprMayStoreToStorage(*_assignment.value, _dialect);
		},
		[&](yul::VariableDeclaration const& _varDecl) -> bool {
			return _varDecl.value && yulExprMayStoreToStorage(*_varDecl.value, _dialect);
		},
		[&](yul::FunctionDefinition const& _funDef) -> bool {
			return yulBlockMayStoreToStorage(_funDef.body, _dialect);
		},
		[&](yul::If const& _if) -> bool {
			return
				(_if.condition && yulExprMayStoreToStorage(*_if.condition, _dialect)) ||
				yulBlockMayStoreToStorage(_if.body, _dialect);
		},
		[&](yul::Switch const& _switch) -> bool {
			if (_switch.expression && yulExprMayStoreToStorage(*_switch.expression, _dialect))
				return true;
			for (auto const& c: _switch.cases)
				if (yulBlockMayStoreToStorage(c.body, _dialect))
					return true;
			return false;
		},
		[&](yul::ForLoop const& _for) -> bool {
			return
				yulBlockMayStoreToStorage(_for.pre, _dialect) ||
				(_for.condition && yulExprMayStoreToStorage(*_for.condition, _dialect)) ||
				yulBlockMayStoreToStorage(_for.post, _dialect) ||
				yulBlockMayStoreToStorage(_for.body, _dialect);
		},
		[&](yul::Break const&) -> bool { return false; },
		[&](yul::Continue const&) -> bool { return false; },
		[&](yul::Leave const&) -> bool { return false; },
		[&](yul::Block const& _block) -> bool {
			return yulBlockMayStoreToStorage(_block, _dialect);
		}
	}, _stmt);
}

// Peel an lvalue/mutated-subexpression down to a base Identifier through
// IndexAccess (a[i]) and MemberAccess (s.field) layers. Returns nullptr when
// the base is not a plain Identifier (e.g. the base is itself a function
// call result, `this`, or some other shape this independent walk does not
// specifically model) — callers must then fail closed (set `unknown`)
// rather than silently dropping the write.
Identifier const* peelToBaseIdentifierForWriteOracle(Expression const& _expr)
{
	Expression const* current = &_expr;
	while (true)
	{
		if (auto const* ident = dynamic_cast<Identifier const*>(current))
			return ident;
		if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(current))
		{
			current = &indexAccess->baseExpression();
			continue;
		}
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(current))
		{
			current = &memberAccess->expression();
			continue;
		}
		return nullptr;
	}
}

// A bounded, well-known set of storage-ref OpenZeppelin library helpers
// (EnumerableSet.add/remove, EnumerableMap.set/remove, Checkpoints.push --
// and their "Upgradeable" twins) whose write target can be soundly
// attributed to the CALL SITE's "self"/"set"/"map" argument, instead of
// giving up with `unknown=true` the way an arbitrary storage-pointer
// parameter must (recordWriteToBase's local-storage-pointer case, just
// below, is and remains the correct fail-closed default for every OTHER
// call -- this function recognizes ONLY this fixed list of (library,
// function) pairs; it is a targeted extension, not a general relaxation
// of that rule). Matched functions are never recursed into: their own
// bodies mutate the "self" struct's fields, which is exactly the
// information their storage-pointer PARAMETER cannot see, so recursing
// would just rediscover the same unresolvable parameter one level down.
bool isKnownOzStorageRefLibraryMutator(FunctionDefinition const& _funcDef)
{
	auto const* library = dynamic_cast<ContractDefinition const*>(_funcDef.scope());
	if (!library || !library->isLibrary())
		return false;
	std::string const& libName = library->name();
	std::string const& fnName = _funcDef.name();
	bool matches =
		((libName == "EnumerableSet" || libName == "EnumerableSetUpgradeable") &&
			(fnName == "add" || fnName == "remove")) ||
		((libName == "EnumerableMap" || libName == "EnumerableMapUpgradeable") &&
			(fnName == "set" || fnName == "remove")) ||
		((libName == "Checkpoints" || libName == "CheckpointsUpgradeable") &&
			fnName == "push");
	if (!matches)
		return false;
	// Defensive shape check: every known signature for these entry points
	// takes the self/set/map argument as a storage-located first
	// parameter. If some unexpected overload doesn't match that shape,
	// fall through to the generic (fail-closed) path instead of guessing.
	auto const& params = _funcDef.parameters();
	if (params.empty())
		return false;
	return params.front()->referenceLocation() == VariableDeclaration::Location::Storage;
}

// Resolve the "self"/"set"/"map" argument of a call already confirmed (via
// isKnownOzStorageRefLibraryMutator) to be one of the bounded OZ helpers,
// covering both calling conventions Solidity allows: using-for member-call
// syntax (`set.add(value)`, where the receiver IS the argument) and
// explicit qualified calls (`EnumerableSet.add(set, value)`, where the
// receiver names the library itself and the argument is the first
// parameter).
Expression const* ozStorageRefSelfArgument(FunctionCall const& _call)
{
	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_call.expression()))
	{
		Type const* receiverType = memberAccess->expression().annotation().type;
		bool isUsingForCall =
			receiverType &&
			receiverType->category() != Type::Category::TypeType &&
			receiverType->category() != Type::Category::Module;
		if (isUsingForCall)
			return &memberAccess->expression();
	}
	if (!_call.arguments().empty())
		return _call.arguments().front().get();
	return nullptr;
}

struct WriteOracleCollector: ASTConstVisitor
{
	std::set<std::string> writes;
	bool unknown = false;
	std::set<FunctionDefinition const*> calleesToVisit;
	ContractDefinition const* mostDerivedContract = nullptr;

	void recordWriteToBase(Expression const& _target)
	{
		if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_target))
		{
			for (auto const& component: tuple->components())
				if (component)
					recordWriteToBase(*component);
			return;
		}

		Identifier const* base = peelToBaseIdentifierForWriteOracle(_target);
		if (!base)
		{
			// Written through something other than a plain (possibly
			// indexed/projected) local/state variable name — e.g. the
			// result of a function call. Fail closed.
			unknown = true;
			return;
		}
		auto const* varDecl = dynamic_cast<VariableDeclaration const*>(base->annotation().referencedDeclaration);
		if (!varDecl)
		{
			// Not a variable reference at all — shouldn't normally happen
			// for a write target, but fail closed rather than dropping it.
			unknown = true;
			return;
		}
		if (varDecl->isStateVariable())
		{
			writes.insert(varDecl->name());
			return;
		}
		if (varDecl->isLocalVariable() && varDecl->referenceLocation() == VariableDeclaration::Location::Storage)
		{
			// A local storage-pointer variable: without alias analysis we
			// cannot statically tell which state variable it points at.
			// Fail closed rather than guessing.
			unknown = true;
			return;
		}
		// Otherwise a plain memory/calldata/stack local: not a storage
		// write at all, nothing to record.
	}

	bool visit(Assignment const& _assignment) override
	{
		recordWriteToBase(_assignment.leftHandSide());
		return true;
	}

	bool visit(UnaryOperation const& _unary) override
	{
		Token op = _unary.getOperator();
		if (op == Token::Delete || op == Token::Inc || op == Token::Dec)
			recordWriteToBase(_unary.subExpression());
		return true;
	}

	bool visit(InlineAssembly const& _asm) override
	{
		if (yulBlockMayStoreToStorage(_asm.operations().root(), _asm.dialect()))
			unknown = true;
		return true;
	}

	bool visit(FunctionCall const& _call) override
	{
		Expression const& calleeExpr = _call.expression();

		// Type conversions and struct-constructor calls are not calls into
		// other code at all — nothing to do.
		if (
			_call.annotation().kind.set() &&
			*_call.annotation().kind != FunctionCallKind::FunctionCall
		)
			return true;

		// Classify by the callee's own FunctionType::Kind rather than by
		// hand-matching identifier/member names: this is the SAME
		// information solc's type checker already computed for every
		// compiler builtin (require/assert/revert/selfdestruct/addmod/
		// keccak256/gasleft/...), low-level call form (call/staticcall/
		// delegatecall/callcode), and push/pop, so it is both more
		// complete and more precise than a name-based allowlist (which
		// would either miss builtins — turning nearly every function with
		// a plain `require(...)` into a false "unknown" — or risk matching
		// a user-defined function that happens to share a builtin's name).
		Type const* calleeType = calleeExpr.annotation().type;
		if (auto const* functionType = dynamic_cast<FunctionType const*>(calleeType))
		{
			using Kind = FunctionType::Kind;
			switch (functionType->kind())
			{
			case Kind::ArrayPush:
			case Kind::ArrayPop:
				// arr.push(...)/arr.pop(): a write to the array's own base
				// state variable (or `unknown` if the base isn't
				// resolvable).
				if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&calleeExpr))
					recordWriteToBase(memberAccess->expression());
				else
					unknown = true;
				return true;
			case Kind::DelegateCall:
			case Kind::BareDelegateCall:
			case Kind::BareCallCode:
				// Executes in OUR storage context under callee-chosen
				// code — cannot be modeled, ever.
				unknown = true;
				return true;
			case Kind::Selfdestruct:
				unknown = true;
				return true;
			case Kind::Internal:
				// Handled below via resolveWriteOracleCallTarget.
				break;
			case Kind::External:
			case Kind::BareCall:
			case Kind::BareStaticCall:
			case Kind::Creation:
			case Kind::Send:
			case Kind::Transfer:
				// Executes in the CALLEE's own storage context (or
				// creates a new contract) — not a direct write to THIS
				// contract's state. (Re-entrancy back into this contract
				// through such a call is a separate, already-modeled
				// concern elsewhere in the pipeline and out of scope for
				// this direct write-set oracle.)
				return true;
			default:
				// Every remaining Kind (KECCAK256/SHA256/RIPEMD160/
				// ECRecover/AddMod/MulMod/BlockHash/BlobHash/GasLeft/
				// Assert/Require/Revert/Error/Event/ABIEncode*/ABIDecode/
				// Wrap/Unwrap/BytesConcat/StringConcat/MetaType/SetGas/
				// SetValue/ERC7201/Declaration/ObjectCreation/...) is a
				// pure or otherwise side-effect-free compiler builtin:
				// never a write to THIS contract's persistent storage.
				return true;
			}
		}

		// No FunctionType (or Kind::Internal): try to resolve the actual
		// FunctionDefinition target and recurse. The worklist in
		// computeAstWriteOracle drains calleesToVisit so this stays
		// iterative, not recursive, for arbitrarily deep call graphs.
		if (FunctionDefinition const* target = resolveWriteOracleCallTarget(calleeExpr, mostDerivedContract))
		{
			if (isKnownOzStorageRefLibraryMutator(*target))
			{
				// Attribute the write directly to the call site's self/set/map
				// argument instead of recursing into the library body (whose
				// OWN storage-pointer parameter cannot see through to it).
				if (Expression const* selfArg = ozStorageRefSelfArgument(_call))
					recordWriteToBase(*selfArg);
				else
					unknown = true;
				return true;
			}
			if (target->isImplemented())
				calleesToVisit.insert(target);
			return true;
		}

		// `new Foo(...)` / `new T[](...)`: contract creation or memory
		// allocation, not a write to OUR persistent storage.
		if (dynamic_cast<NewExpression const*>(&calleeExpr))
			return true;

		// Anything else is a call whose target we could not positively
		// identify — most notably an indirect call through a function-
		// typed local/parameter/array element. Fail closed.
		unknown = true;
		return true;
	}
};

// Compute the AST write-set oracle for a single exported function, over the
// transitive closure of internal/library/super/virtual calls AND the
// modifier chain that actually runs whenever this function is invoked.
// Mirrors collectSuperReferencedFunctions()'s worklist idiom (a visited set
// plus an explicit std::vector worklist draining to a fixpoint) rather than
// recursing directly, so this scales to arbitrarily deep call graphs. Must
// never throw: anything unexpected is folded into `unknown`.
WriteOracleResult computeAstWriteOracle(
	FunctionDefinition const& _function,
	ContractDefinition const& _contract)
{
	WriteOracleResult result;
	if (!_function.isImplemented())
		return result;

	std::set<FunctionDefinition const*> visitedFunctions;
	std::set<ModifierDefinition const*> visitedModifiers;
	std::vector<FunctionDefinition const*> functionWorklist{ &_function };
	std::vector<ModifierDefinition const*> modifierWorklist;

	try
	{
		while (!functionWorklist.empty() || !modifierWorklist.empty())
		{
			if (!functionWorklist.empty())
			{
				FunctionDefinition const* fn = functionWorklist.back();
				functionWorklist.pop_back();
				if (!fn || visitedFunctions.count(fn) || !fn->isImplemented())
					continue;
				visitedFunctions.insert(fn);

				WriteOracleCollector collector;
				collector.mostDerivedContract = &_contract;
				fn->body().accept(collector);
				result.writes.insert(collector.writes.begin(), collector.writes.end());
				result.unknown = result.unknown || collector.unknown;
				for (FunctionDefinition const* callee: collector.calleesToVisit)
					if (!visitedFunctions.count(callee))
						functionWorklist.push_back(callee);

				for (auto const& modifierInvocation: fn->modifiers())
				{
					ModifierDefinition const* modDef =
						resolveModifierDefinition(*fn, *modifierInvocation);
					if (modDef && modDef->isImplemented() && !visitedModifiers.count(modDef))
						modifierWorklist.push_back(modDef);
				}
				continue;
			}

			ModifierDefinition const* mod = modifierWorklist.back();
			modifierWorklist.pop_back();
			if (!mod || visitedModifiers.count(mod) || !mod->isImplemented())
				continue;
			visitedModifiers.insert(mod);

			WriteOracleCollector collector;
			collector.mostDerivedContract = &_contract;
			mod->body().accept(collector);
			result.writes.insert(collector.writes.begin(), collector.writes.end());
			result.unknown = result.unknown || collector.unknown;
			for (FunctionDefinition const* callee: collector.calleesToVisit)
				if (!visitedFunctions.count(callee))
					functionWorklist.push_back(callee);
		}
	}
	catch (...)
	{
		// The oracle must never throw. If something truly unexpected
		// happens mid-walk, the safe answer is "we don't know", not a
		// crash and not a silently-empty write-set.
		result.unknown = true;
	}

	return result;
}

Json astWriteOracleJson(FunctionDefinition const& _function, ContractDefinition const& _contract)
{
	WriteOracleResult oracle = computeAstWriteOracle(_function, _contract);
	Json result = Json::object();
	result["writes"] = Json::array();
	for (std::string const& name: oracle.writes)
		result["writes"].emplace_back(name);
	result["unknown"] = oracle.unknown;
	return result;
}

Json exportBody(FunctionDefinition const& _function)
{
	// Save and clear per-function namespaced storage aliases
	// (the getter prefix map persists across functions)
	auto savedAliases = namespacedStorageAliases;
	namespacedStorageAliases.clear();

	Json body;
	try
	{
		body = exportStmt(_function.body());
		body = expandModifiers(_function, std::move(body));
	}
	catch (...)
	{
		namespacedStorageAliases = savedAliases;
		// Capture a diagnostic reason where we can (UnsupportedSolCore
		// carries a human-readable message; anything else is opaque).
		std::string reason = "unknown export failure";
		try
		{
			throw;
		}
		catch (UnsupportedSolCore const& e)
		{
			reason = e.what();
		}
		catch (std::exception const& e)
		{
			reason = e.what();
		}
		catch (...)
		{
			// Truly unknown throw (e.g. not derived from std::exception) —
			// keep the generic reason string above.
		}
		// Deliberately NOT a valid "block"/"statements" shape (the shape a
		// well-formed function body always has). Substituting a normal
		// no-op block here would let ANY consumer of this JSON — including
		// ones that don't know about this exporter's failure modes —
		// silently accept a wrong-but-plausible empty function. Instead,
		// surface a body value that fails to parse for every consumer: our
		// own OCaml frontend (see SolCoreOfJson.ml's parse_body) raises a
		// hard parse error the moment it sees "kind":"unsupported_body",
		// and any other tool trying to read "statements" off this object
		// will find it missing rather than (wrongly) empty.
		Json failedBody = Json::object();
		failedBody["kind"] = "unsupported_body";
		failedBody["error"] = reason;
		return failedBody;
	}
	if (!body.is_object() || body.value("kind", ""s) != "block")
	{
		namespacedStorageAliases = savedAliases;
		return body;
	}

	// Prepend implicit local variable declarations for named return parameters.
	// In Solidity, named return parameters like "returns (uint256 y)" create an
	// implicit local variable "y" that is initialized to the type's default value.
	if (!_function.returnParameters().empty())
	{
		for (auto const& retParam: _function.returnParameters())
		{
			if (!retParam->name().empty())
			{
				Json letStmt = Json::object();
				letStmt["kind"] = "let";
				letStmt["name"] = retParam->name();
				try
				{
					letStmt["type"] = exportTypeName(retParam->typeName());
					letStmt["value"] = defaultValueForTypeName(retParam->typeName());
				}
				catch (...)
				{
					letStmt["type"] = Json("u256");
					Json zero = Json::object();
					zero["kind"] = "u256";
					zero["value"] = "0";
					letStmt["value"] = zero;
				}
				// Prepend at the beginning of the block
				Json newStatements = Json::array();
				newStatements.emplace_back(letStmt);
				for (auto const& stmt: body["statements"])
					newStatements.emplace_back(stmt);
				body["statements"] = newStatements;
			}
		}
	}

	// Ensure function body ends with an explicit return
	{
		Json const& statements = body["statements"];
		bool hasExplicitReturn = false;
		for (auto const& stmt: statements)
			if (stmt.is_object() && stmt.value("kind", ""s) == "return")
				hasExplicitReturn = true;
		if (!hasExplicitReturn)
		{
			Json ret = Json::object();
			ret["kind"] = "return";
			if (_function.returnParameters().empty())
			{
				Json unit = Json::object();
				unit["kind"] = "unit";
				ret["value"] = unit;
			}
			else if (_function.returnParameters().size() == 1
			         && !_function.returnParameters().front()->name().empty())
			{
				// Implicit return of the named return variable
				Json local = Json::object();
				local["kind"] = "local";
				local["name"] = _function.returnParameters().front()->name();
				ret["value"] = local;
			}
			else if (_function.returnParameters().size() == 1)
			{
				// Unnamed return parameter — return default value for the type
				ret["value"] = defaultValueForTypeName(_function.returnParameters().front()->typeName());
			}
			else
			{
				Json unit = Json::object();
				unit["kind"] = "unit";
				ret["value"] = unit;
			}
			body["statements"].emplace_back(ret);
		}
	}
	namespacedStorageAliases = savedAliases;
	return body;
}

Json exportFunction(FunctionDefinition const& _function, ContractDefinition const& _contract, bool _isInternal = false)
{
	if (!_function.isOrdinary() || !_function.isImplemented())
		throw UnsupportedSolCore("Only ordinary implemented functions are supported.");
	if (!_isInternal && !(_function.visibility() == Visibility::Public || _function.visibility() == Visibility::External))
		throw UnsupportedSolCore("Only public/external functions are supported.");
	// Note: multiple return values are exported as a tuple return type.

	Json result = Json::object();
	result["name"] = exportedFunctionName(_function);
	if (result["name"] != (_function.name().empty() ? "_unnamed" : _function.name()))
		result["originalName"] = _function.name().empty() ? "_unnamed" : _function.name();
	if (_isInternal)
		result["visibility"] = "internal";
	if (!_function.modifiers().empty())
		result["has_modifiers"] = true;
	result["params"] = Json::array();
	for (auto const& parameter: _function.parameters())
	{
		try
		{
			result["params"].emplace_back(exportParam(*parameter));
		}
		catch (...)
		{
			// If a parameter type is unsupported, emit it as u256
			Json param = Json::object();
			param["name"] = parameter->name().empty() ? ("arg" + std::to_string(parameter->id())) : parameter->name();
			param["type"] = Json("u256");
			result["params"].emplace_back(param);
		}
	}
	if (_function.returnParameters().empty())
		result["return"] = Json("unit");
	else if (_function.returnParameters().size() == 1)
	{
		try
		{
			result["return"] = exportTypeName(_function.returnParameters().front()->typeName());
		}
		catch (...)
		{
			result["return"] = Json("u256");
		}
	}
	else
	{
		// Multiple return values — export as tuple type
		Json tupleType = Json::object();
		tupleType["kind"] = "tuple";
		tupleType["elements"] = Json::array();
		for (auto const& retParam: _function.returnParameters())
		{
			try
			{
				tupleType["elements"].emplace_back(exportTypeName(retParam->typeName()));
			}
			catch (...)
			{
				tupleType["elements"].emplace_back(Json("u256"));
			}
		}
		result["return"] = tupleType;
	}
	result["body"] = exportBody(_function);
	// Layer 2 (SOL-PLAN-FIDELITY): independent AST write-set cross-check.
	// Computed even when the body export above succeeded — it is a second
	// opinion, not just a failure fallback.
	result["ast_write_oracle"] = astWriteOracleJson(_function, _contract);
	return result;
}

Json exportConstructor(FunctionDefinition const& _function, ContractDefinition const& _contract)
{
	if (_function.isOrdinary() || !_function.isImplemented())
		throw UnsupportedSolCore("Only implemented constructors are supported.");

	Json result = Json::object();
	result["name"] = "constructor";
	result["params"] = Json::array();
	for (auto const& parameter: _function.parameters())
	{
		try
		{
			result["params"].emplace_back(exportParam(*parameter));
		}
		catch (...)
		{
			Json param = Json::object();
			param["name"] = parameter->name().empty() ? ("arg" + std::to_string(parameter->id())) : parameter->name();
			param["type"] = Json("u256");
			result["params"].emplace_back(param);
		}
	}
	result["return"] = Json("unit");
	result["body"] = exportBody(_function);
	result["ast_write_oracle"] = astWriteOracleJson(_function, _contract);
	return result;
}

Json exportOrigins(ContractDefinition const& _contract)
{
	Json result = Json::object();
	result["entries"] = Json::array();
	unsigned slotIndex = 0;
	for (VariableDeclaration const* stateVar: _contract.stateVariables())
	{
		// Skip constant and immutable variables — they don't have storage slots
		if (stateVar->isConstant() || stateVar->immutable())
			continue;
		Json entry = Json::object();
		entry["originId"] = "state:" + stateVar->name();
		// SOLCORE_TLOAD_TSTORE_ARCHITECTURE_SPEC.md §8 Stage 4 gate: mark
		// `transient`-location state variables distinctly rather than
		// silently tagging them "stateVariable" (which would misleadingly
		// imply an ordinary persistent slot). The primary fail-closed gate
		// for this keyword lives in `exportStateVars` above (which throws
		// `UnsupportedSolCore` and fails the whole contract's export before
		// this function even runs, in the only pipeline that currently
		// calls both) — this is defense-in-depth for `exportOrigins`
		// specifically, since it is pure provenance/audit metadata, not the
		// semantic `Storage` model itself, so "mark it" (not "fail") is the
		// right-sized response here per the spec's own "either mark it in
		// JSON or fail the export" phrasing.
		entry["kind"] =
			stateVar->referenceLocation() == VariableDeclaration::Location::Transient
				? "stateVariableTransient"
				: "stateVariable";
		entry["astId"] = stateVar->id();
		entry["name"] = stateVar->name();
		entry["storageSlot"] = slotIndex;
		entry["sourceLocation"] = sourceLocation(stateVar->location());
		result["entries"].emplace_back(entry);
		++slotIndex;
	}
	for (FunctionDefinition const* function: _contract.definedFunctions())
	{
		if (!function->isOrdinary())
			continue;
		Json entry = Json::object();
		// externalSignature() can throw for internal functions with storage reference parameters
		try
		{
			entry["originId"] = "function:" + function->externalSignature();
			entry["signature"] = function->externalSignature();
		}
		catch (...)
		{
			entry["originId"] = "function:" + function->name();
			entry["signature"] = function->name();
		}
		entry["kind"] = "function";
		entry["astId"] = function->id();
		entry["name"] = function->name();
		Json paramTypes = Json::array();
		for (auto const& parameter: function->parameters())
		{
			try
			{
				paramTypes.emplace_back(exportTypeName(parameter->typeName()));
			}
			catch (UnsupportedSolCore const&)
			{
				paramTypes.emplace_back(Json("unknown"));
			}
			catch (...)
			{
				paramTypes.emplace_back(Json("unknown"));
			}
		}
		entry["parameterTypes"] = std::move(paramTypes);
		entry["sourceLocation"] = sourceLocation(function->location());
		result["entries"].emplace_back(entry);
	}
	// Add origin entries for receive/fallback if they exist
	if (auto const* recv = _contract.receiveFunction())
	{
		Json entry = Json::object();
		entry["originId"] = "function:receive()";
		entry["kind"] = "function";
		entry["astId"] = recv->id();
		entry["name"] = "receive";
		entry["signature"] = "receive()";
		entry["parameterTypes"] = Json::array();
		entry["sourceLocation"] = sourceLocation(recv->location());
		result["entries"].emplace_back(entry);
	}
	if (auto const* fb = _contract.fallbackFunction())
	{
		Json entry = Json::object();
		entry["originId"] = "function:fallback()";
		entry["kind"] = "function";
		entry["astId"] = fb->id();
		entry["name"] = "fallback";
		entry["signature"] = "fallback()";
		entry["parameterTypes"] = Json::array();
		entry["sourceLocation"] = sourceLocation(fb->location());
		result["entries"].emplace_back(entry);
	}
	for (EventDefinition const* event: _contract.events())
	{
		Json entry = Json::object();
		entry["originId"] = "event:" + event->name();
		entry["kind"] = "event";
		entry["astId"] = event->id();
		entry["name"] = event->name();
		entry["anonymous"] = event->isAnonymous();
		Json paramTypes = Json::array();
		for (auto const& parameter: event->parameters())
		{
			try
			{
				Json paramInfo = Json::object();
				paramInfo["type"] = exportTypeName(parameter->typeName());
				paramInfo["indexed"] = parameter->isIndexed();
				paramTypes.emplace_back(std::move(paramInfo));
			}
			catch (UnsupportedSolCore const&)
			{
				Json paramInfo = Json::object();
				paramInfo["type"] = Json("unknown");
				paramInfo["indexed"] = parameter->isIndexed();
				paramTypes.emplace_back(std::move(paramInfo));
			}
		}
		entry["parameterTypes"] = std::move(paramTypes);
		entry["sourceLocation"] = sourceLocation(event->location());
		result["entries"].emplace_back(entry);
	}
	return result;
}

Json runtimeTypeDecl(std::string const& _name, Json _fields)
{
	Json decl = Json::object();
	decl["name"] = _name;
	decl["fields"] = std::move(_fields);
	return decl;
}

Json exportSourceImports(SourceUnit const& _sourceUnit)
{
	Json imports = Json::array();
	std::set<std::string> seen;
	for (auto const& node: _sourceUnit.nodes())
	{
		auto const* importDirective = dynamic_cast<ImportDirective const*>(node.get());
		if (!importDirective)
			continue;
		std::string sourcePath =
			importDirective->annotation().absolutePath.set() ?
			*importDirective->annotation().absolutePath :
			std::string{};
		if (sourcePath.empty())
			sourcePath = importDirective->path();
		if (sourcePath.empty() || !seen.insert(sourcePath).second)
			continue;
		imports.emplace_back(sourcePath);
	}
	return imports;
}

namespace v0_8
{

solcore::ExportArtifacts exportContract(CompilerStack const& _compilerStack, std::string const& _contractName)
{
	struct ActiveCompilerStackScope
	{
		CompilerStack const* saved = activeCompilerStack;
		ContractDefinition const* savedContract = activeExportContract;
		explicit ActiveCompilerStackScope(CompilerStack const& _compilerStack)
		{
			activeCompilerStack = &_compilerStack;
		}
		~ActiveCompilerStackScope()
		{
			activeCompilerStack = saved;
			activeExportContract = savedContract;
		}
	} activeCompilerStackScope(_compilerStack);

	ContractDefinition const& contract = _compilerStack.contractDefinition(_contractName);
	activeExportContract = &contract;

	// --- Detect namespaced storage getters (ERC-7201 pattern) ---
	// Clear thread-local state from any previous export
	namespacedStorageAliases.clear();
	namespacedGetterPrefixes.clear();
	namespacedMappingFields.clear();
	exportedFunctionNames.clear();

	std::vector<NamespacedStorageGetter> namespacedGetters;
	// Scan all functions in the contract hierarchy for namespaced storage getters
	for (FunctionDefinition const* function : contract.definedFunctions())
	{
		StructDefinition const* structDef = nullptr;
		if (isNamespacedStorageGetter(*function, &structDef))
		{
			std::string prefix = derivePrefix(structDef->name());
			std::string fieldName = deriveSubStorageFieldName(structDef->name());
			namespacedGetters.push_back({function, structDef, prefix, fieldName});
			namespacedGetterPrefixes[function] = prefix;
			// Track which prefixed fields are mappings
			for (auto const& member : structDef->members())
			{
				if (dynamic_cast<Mapping const*>(&member->typeName()))
					namespacedMappingFields.insert(prefix + member->name());
			}
		}
	}
	// Also scan base contracts
	for (auto const* baseContract : contract.annotation().linearizedBaseContracts)
	{
		if (baseContract == &contract)
			continue;
		for (FunctionDefinition const* function : baseContract->definedFunctions())
		{
			StructDefinition const* structDef = nullptr;
			if (isNamespacedStorageGetter(*function, &structDef))
			{
				// Avoid duplicates
				bool alreadyFound = false;
				for (auto const& g : namespacedGetters)
					if (g.function->name() == function->name())
						alreadyFound = true;
				if (!alreadyFound)
				{
					std::string prefix = derivePrefix(structDef->name());
					std::string fieldName = deriveSubStorageFieldName(structDef->name());
					namespacedGetters.push_back({function, structDef, prefix, fieldName});
					namespacedGetterPrefixes[function] = prefix;
					for (auto const& member : structDef->members())
					{
						if (dynamic_cast<Mapping const*>(&member->typeName()))
							namespacedMappingFields.insert(prefix + member->name());
					}
				}
			}
		}
	}
	// --- End namespaced storage getter detection ---

	Json solcore = Json::object();
	// 0.2.0: hardened body-export-failure marker ("unsupported_body" instead
	// of a silently-valid no-op block) + per-function "ast_write_oracle"
	// (independent AST write-set cross-check; see computeAstWriteOracle).
	solcore["solcoreVersion"] = "0.2.0";
	solcore["solidityVersion"] = VersionString;
	solcore["featureFlags"] = featureFlags();
	Json metadata = exporterMetadata(_contractName);
	for (auto const& [key, value]: metadata.items())
		solcore[key] = value;
	solcore["crate_name"] = contract.name();
	solcore["deployedCodeSize"] = solcoreDeployedCodeSize(contract);
	solcore["deployedCodeSizeMode"] = "semantic-nonzero";
	solcore["source_imports"] = exportSourceImports(contract.sourceUnit());

	Json typeDecls = Json::array();
	Json subStorageGetters = Json::array();
	Json storageFields = Json::array();
	std::set<std::string> exportedFieldNames;

	// Helper lambda to export state variables from a contract definition
	auto exportStateVars = [&](ContractDefinition const* source)
	{
		for (VariableDeclaration const* stateVar: source->stateVariables())
		{
			// Skip constant variables — they are inlined at usage sites
			if (stateVar->isConstant())
				continue;
			// SOLCORE_TLOAD_TSTORE_ARCHITECTURE_SPEC.md §8 Stage 4 gate: a
			// `transient`-location state variable (solc 0.8.28+,
			// `uint256 transient x;`) is a DIFFERENT feature from the raw
			// `tload`/`tstore` Yul builtins Stage 1 models — it has its own
			// (currently unimplemented) reset/slot semantics. Silently
			// falling through to the code below would export it exactly
			// like an ordinary persistent state variable: a false model
			// (wrong reset semantics, wrong slot space) that every
			// downstream `Storage`-preservation theorem would then be
			// "proving" about the wrong thing. No corpus contract uses this
			// keyword today (OZ's transient-storage helpers stay on raw
			// assembly for pre-0.8.28 compatibility), so failing the whole
			// contract's export now is cheap and exactly matches this
			// exporter's existing `UnsupportedSolCore` idiom — propagates to
			// `exportContract`'s outer catch, which turns it into a
			// structured `{"unsupported": true, "reason": ...}` export
			// rather than a silently-dropped field. Fail-closed beats wrong.
			if (stateVar->referenceLocation() == VariableDeclaration::Location::Transient)
				throw UnsupportedSolCore(
					"`transient`-location state variable '" + stateVar->name() +
					"' is not yet modeled (EIP-1153 transient storage is only "
					"supported via raw tload/tstore Yul builtins in inline "
					"assembly, not the `transient` state-variable keyword); "
					"exporting it as an ordinary persistent field would be a "
					"false model."
				);
			// Skip already-exported fields (can happen with diamond inheritance)
			if (exportedFieldNames.count(stateVar->name()))
				continue;
			// Skip immutable variables whose value can be successfully inlined.
			if (stateVar->immutable() && stateVar->value())
			{
				try
				{
					(void)exportExpr(*stateVar->value());
					continue;
				}
				catch (...)
				{
				}
			}
			try
			{
				storageFields.emplace_back(exportStorageField(_compilerStack, contract, *stateVar));
				exportedFieldNames.insert(stateVar->name());
			}
			catch (UnsupportedSolCore const&)
			{
			}
			catch (std::exception const&)
			{
			}
		}
	};

	// Export state variables from the full inheritance hierarchy
	// Iterate in linearization order (most-base first) so storage layout matches
	{
		auto const& bases = contract.annotation().linearizedBaseContracts;
		for (auto it = bases.rbegin(); it != bases.rend(); ++it)
			exportStateVars(*it);
	}

	// --- Flatten namespaced storage struct fields into Storage ---
	for (auto const& getter : namespacedGetters)
	{
		for (auto const& member : getter.structDef->members())
		{
			std::string flatName = getter.prefix + member->name();
			// Skip if we already exported a field with this name
			if (exportedFieldNames.count(flatName))
				continue;
			try
			{
				Json field = Json::object();
				field["name"] = flatName;
				field["type"] = exportTypeName(member->typeName());
				storageFields.emplace_back(field);
				exportedFieldNames.insert(flatName);
			}
			catch (UnsupportedSolCore const&)
			{
				// Skip fields with unsupported types
			}
			catch (std::exception const&)
			{
				// Skip fields that cause unexpected errors
			}
		}
	}
	// --- End flatten namespaced storage ---

	typeDecls.emplace_back(runtimeTypeDecl("Storage", std::move(storageFields)));

	Json callEnvFields = Json::array();
	callEnvFields.emplace_back(Json{{"name", "msgSender"}, {"type", "address"}});
	callEnvFields.emplace_back(Json{{"name", "msgValue"}, {"type", "u256"}});
	callEnvFields.emplace_back(Json{{"name", "blockTimestamp"}, {"type", "u256"}});
	callEnvFields.emplace_back(Json{{"name", "blockNumber"}, {"type", "u256"}});
	callEnvFields.emplace_back(Json{{"name", "chainId"}, {"type", "u256"}});
	callEnvFields.emplace_back(Json{{"name", "thisAddress"}, {"type", "address"}});
	{
		Json calldataType = Json::object();
		calldataType["kind"] = "array";
		calldataType["element"] = "u8";
		callEnvFields.emplace_back(Json{{"name", "calldata"}, {"type", calldataType}});
	}
	typeDecls.emplace_back(runtimeTypeDecl("CallEnv", std::move(callEnvFields)));
	Json worldStateFields = Json::array();
	worldStateFields.emplace_back(Json{{"name", "contractBalance"}, {"type", "u256"}});
	worldStateFields.emplace_back(Json{{"name", "codeSize"}, {"type", Json{
		{"kind", "mapping"},
		{"key", "address"},
		{"value", "u256"}
	}}});
	worldStateFields.emplace_back(Json{{"name", "contractStorage"}, {"type", Json{
		{"kind", "mapping"},
		{"key", "address"},
		{"value", Json{
			{"kind", "mapping"},
			{"key", "u256"},
			{"value", "u256"}
		}}
	}}});
	typeDecls.emplace_back(runtimeTypeDecl("WorldState", std::move(worldStateFields)));
	typeDecls.emplace_back(runtimeTypeDecl("Memory", Json::array()));
	typeDecls.emplace_back(runtimeTypeDecl("ByteArray", Json::array()));
	typeDecls.emplace_back(runtimeTypeDecl("Logs", Json::array()));

	// Export user-defined struct types
	// Check top-level structs in the source unit
	for (auto const& node : contract.sourceUnit().nodes())
	{
		if (auto const* structDef = dynamic_cast<StructDefinition const*>(node.get()))
		{
			Json structFields = Json::array();
			for (auto const& member : structDef->members())
			{
				try
				{
					structFields.emplace_back(exportField(*member));
				}
				catch (UnsupportedSolCore const&)
				{
					// Skip struct fields with unsupported types
				}
				catch (std::exception const&)
				{
					// Skip struct fields that cause unexpected errors during export
				}
			}
			typeDecls.emplace_back(runtimeTypeDecl(structDef->name(), std::move(structFields)));
		}
	}
	// Also check structs defined inside contracts and interfaces
	for (auto const& node : contract.sourceUnit().nodes())
	{
		if (auto const* contractNode = dynamic_cast<ContractDefinition const*>(node.get()))
		{
			for (auto const* structDef : contractNode->definedStructs())
			{
				// Avoid duplicates (struct might already be added from source unit level)
				bool exists = false;
				for (auto const& existing : typeDecls)
					if (existing.value("name", "") == structDef->name())
						exists = true;
				if (!exists)
				{
					Json structFields = Json::array();
					for (auto const& member : structDef->members())
					{
						try
						{
							structFields.emplace_back(exportField(*member));
						}
						catch (UnsupportedSolCore const&)
						{
							// Skip struct fields with unsupported types
						}
					}
					typeDecls.emplace_back(runtimeTypeDecl(structDef->name(), std::move(structFields)));
				}
			}
		}
	}

	// Also export struct types from all reachable source units — including
	// base contracts, imported libraries, and transitively imported files.
	{
		std::set<std::string> addedTypeNames;
		for (auto const& existing : typeDecls)
			addedTypeNames.insert(existing.value("name", ""));

		auto tryAddStruct = [&](StructDefinition const* structDef)
		{
			if (addedTypeNames.count(structDef->name()))
				return;
			Json structFields = Json::array();
			for (auto const& member : structDef->members())
			{
				try
				{
					structFields.emplace_back(exportField(*member));
				}
				catch (...)
				{
				}
			}
			typeDecls.emplace_back(runtimeTypeDecl(structDef->name(), std::move(structFields)));
			addedTypeNames.insert(structDef->name());
		};

		// Collect all reachable source units by walking the import graph
		std::set<SourceUnit const*> visitedUnits;
		std::vector<SourceUnit const*> unitQueue;
		unitQueue.push_back(&contract.sourceUnit());
		for (auto const* baseContract : contract.annotation().linearizedBaseContracts)
			unitQueue.push_back(&baseContract->sourceUnit());

		while (!unitQueue.empty())
		{
			SourceUnit const* unit = unitQueue.back();
			unitQueue.pop_back();
			if (!visitedUnits.insert(unit).second)
				continue;

			// Scan this source unit for struct definitions
			for (auto const& node : unit->nodes())
			{
				if (auto const* structDef = dynamic_cast<StructDefinition const*>(node.get()))
					tryAddStruct(structDef);
				if (auto const* contractNode = dynamic_cast<ContractDefinition const*>(node.get()))
					for (auto const* structDef : contractNode->definedStructs())
						tryAddStruct(structDef);
				// Follow imports to reach transitively imported source units
				if (auto const* importDir = dynamic_cast<ImportDirective const*>(node.get()))
					if (importDir->annotation().sourceUnit)
						unitQueue.push_back(importDir->annotation().sourceUnit);
			}
		}
	}

	solcore["type_decls"] = std::move(typeDecls);
	solcore["sub_storage_getters"] = std::move(subStorageGetters);
	Json foreignContracts = Json::array();
	for (std::string const& candidate: _compilerStack.contractNames())
	{
		try
		{
			foreignContracts.emplace_back(
				exportForeignContractSummary(_compilerStack, candidate)
			);
		}
		catch (...)
		{
		}
	}
	solcore["foreign_contracts"] = std::move(foreignContracts);

	Json state = Json::object();
	state["name"] = "ExecState";
	Json stateFields = Json::array();
	stateFields.emplace_back(Json{{"name", "world"}, {"type", Json{{"kind", "named"}, {"name", "WorldState"}}}});
	stateFields.emplace_back(Json{{"name", "storage"}, {"type", Json{{"kind", "named"}, {"name", "Storage"}}}});
	stateFields.emplace_back(Json{{"name", "memory"}, {"type", Json{{"kind", "named"}, {"name", "Memory"}}}});
	stateFields.emplace_back(Json{{"name", "returndata"}, {"type", Json{{"kind", "named"}, {"name", "ByteArray"}}}});
	stateFields.emplace_back(Json{{"name", "logs"}, {"type", Json{{"kind", "named"}, {"name", "Logs"}}}});
	stateFields.emplace_back(Json{{"name", "env"}, {"type", Json{{"kind", "named"}, {"name", "CallEnv"}}}});
	// SOLCORE_TLOAD_TSTORE_ARCHITECTURE_SPEC.md §3.1: deliberately NOT
	// listing the new `transient` ExecState field here. This list is not
	// purely informational/inert as originally assumed while implementing
	// Stage 1 — SolCoreOfJson.ml's `parse_state_decl` feeds it into
	// `type_env.state.fields`, which downstream code keys named lookups off
	// of (e.g. LeanSupport.ml's `storage_record_context`), and adding an
	// entry with a `type.name` ("TransientStorage") that has no registered
	// SolCore named-type mapping broke real-corpus generation (verified:
	// `state.transient` mis-typed to `U256` instead of `TransientStorage`,
	// a Lean compile failure on MetricReentrancyGuardTransient). The
	// `transient` field is real and fully wired at the runtime-type level
	// (Types.lean/Mapping.lean) and in the Yul translation
	// (Base.ml's tload/tstore dispatch); it does not need a `state.fields`
	// entry for either of those to work correctly.
	state["fields"] = std::move(stateFields);
	solcore["state"] = std::move(state);

	// Assign export names (and register `f__super__Owner` aliases for every
	// statically-bound base-call target that needs flattening) BEFORE
	// exporting the constructor and dispatch entries below: both lower call
	// names via exportedFunctionName, which previously ran against a stale
	// map (whatever the previously exported contract left behind — empty for
	// the first contract), so a constructor-body `super.f()`/`Base.f()` call
	// or a cross-contract overload could get a name inconsistent with the
	// sibling definitions this same artifact exports further below.
	std::set<FunctionDefinition const*> namespacedGetterDefinitions;
	for (auto const& getter : namespacedGetters)
		namespacedGetterDefinitions.insert(getter.function);

	std::vector<FunctionDefinition const*> overloadCandidates;
	for (FunctionDefinition const* function : contract.definedFunctions())
	{
		if (!function->isOrdinary() || !function->isImplemented())
			continue;
		if (namespacedGetterDefinitions.count(function))
			continue;
		overloadCandidates.push_back(function);
	}
	for (auto const* baseContract : contract.annotation().linearizedBaseContracts)
	{
		if (baseContract == &contract)
			continue;
		for (FunctionDefinition const* function : baseContract->definedFunctions())
		{
			if (!function->isOrdinary() || !function->isImplemented())
				continue;
			if (namespacedGetterDefinitions.count(function))
				continue;
			overloadCandidates.push_back(function);
		}
	}
	assignExportedFunctionNames(overloadCandidates);
	std::set<FunctionDefinition const*> superReferencedFunctions =
		collectSuperReferencedFunctions(contract);
	for (FunctionDefinition const* function: superReferencedFunctions)
		exportedFunctionNames[function] = contractScopedSuperAlias(*function);

	if (auto const* ctor = contract.constructor())
	{
		if (ctor->isImplemented())
		{
			try
			{
				solcore["constructor"] = exportConstructor(*ctor, contract);
			}
			catch (UnsupportedSolCore const&)
			{
				// Skip unsupported constructors
			}
			catch (std::exception const&)
			{
				// Skip constructors that cause unexpected export errors
			}
		}
	}

	Json dispatchEntries = Json::array();
	try
	{
		for (auto const& [selector, functionType]: contract.interfaceFunctions())
		{
			(void)selector;
			if (!functionType)
				continue;
			dispatchEntries.emplace_back(
				exportDispatchEntry(
					functionType,
					dynamic_cast<FunctionDefinition const*>(&functionType->declaration())
				)
			);
		}
	}
	catch (...)
	{
	}
	solcore["dispatch_entries"] = std::move(dispatchEntries);

	Json functions = Json::array();
	Json internalFunctions = Json::array();

	// Collect exported internal function names to avoid duplicates
	std::set<std::string> exportedInternalNames;

	subStorageGetters = Json::array();
	for (auto const& getter : namespacedGetters)
	{
		Json entry = Json::object();
		entry["function"] = exportedFunctionName(*getter.function);
		entry["field"] = getter.fieldName;
		entry["type"] = getter.structDef->name();
		subStorageGetters.emplace_back(std::move(entry));
	}

	// First pass: export functions defined directly in this contract
	for (FunctionDefinition const* function: contract.definedFunctions())
	{
		if (!function->isOrdinary() || !function->isImplemented())
			continue;
		// Skip namespaced storage getter functions — they are not real functions
		if (namespacedGetterDefinitions.count(function))
			continue;
		if (function->visibility() == Visibility::Public || function->visibility() == Visibility::External)
		{
			try
			{
				functions.emplace_back(exportFunction(*function, contract));
			}
			catch (UnsupportedSolCore const&)
			{
				// Skip unsupported public functions
			}
			catch (std::exception const&)
			{
				// Skip functions that cause unexpected errors during export
			}
		}
		else
		{
			// Internal/private helper functions
			try
			{
				internalFunctions.emplace_back(exportFunction(*function, contract, /*_isInternal=*/true));
			}
			catch (UnsupportedSolCore const&)
			{
				// Skip unsupported internal functions — they may use features not yet exported
			}
			catch (std::exception const&)
			{
				// Skip internal functions that cause unexpected errors during export
			}
		}
	}

	// Track which internal function names we already exported
	for (auto const& f : internalFunctions)
		if (f.contains("name") && f["name"].is_string())
			exportedInternalNames.insert(f["name"].get<std::string>());

	// Track which public function names we already exported
	std::set<std::string> exportedPublicNames;
	for (auto const& f : functions)
		if (f.contains("name") && f["name"].is_string())
			exportedPublicNames.insert(f["name"].get<std::string>());

	// Second pass: export inherited public functions from base contracts,
	// iterating in reverse linearization order (most-base first) so that
	// actual implementations are preferred over super-delegating overrides.
	{
		auto const& bases = contract.annotation().linearizedBaseContracts;
		for (auto it = bases.rbegin(); it != bases.rend(); ++it)
		{
			auto const* baseContract = *it;
			if (baseContract == &contract)
				continue;
			for (FunctionDefinition const* function : baseContract->definedFunctions())
			{
				if (!function->isOrdinary() || !function->isImplemented())
					continue;
				if (namespacedGetterDefinitions.count(function))
					continue;
				if (function->visibility() != Visibility::Public && function->visibility() != Visibility::External)
					continue;
				if (superReferencedFunctions.count(function))
					continue;
				if (exportedPublicNames.count(exportedFunctionName(*function)))
					continue;
				try
				{
					functions.emplace_back(exportFunction(*function, contract));
					exportedPublicNames.insert(exportedFunctionName(*function));
				}
				catch (UnsupportedSolCore const&) {}
				catch (std::exception const&) {}
			}
		}
	}

	// Third pass: export inherited internal functions from base contracts
	// (most-derived first so overrides win)
	for (auto const* baseContract : contract.annotation().linearizedBaseContracts)
	{
		if (baseContract == &contract)
			continue; // Skip self
		for (FunctionDefinition const* function : baseContract->definedFunctions())
		{
			if (!function->isOrdinary() || !function->isImplemented())
				continue;
			// Skip namespaced storage getter functions
			if (namespacedGetterDefinitions.count(function))
				continue;
			bool needsSuperAlias = superReferencedFunctions.count(function);
			if (
				(function->visibility() == Visibility::Public || function->visibility() == Visibility::External) &&
				!needsSuperAlias
			)
				continue; // Already handled in public pass above
			if (exportedInternalNames.count(exportedFunctionName(*function)))
				continue; // Already exported (overridden in derived contract)
			try
			{
				internalFunctions.emplace_back(exportFunction(*function, contract, /*_isInternal=*/true));
				exportedInternalNames.insert(exportedFunctionName(*function));
			}
			catch (UnsupportedSolCore const&) {}
			catch (std::exception const&) {}
		}
	}

	// Export receive() function if present
	if (auto const* recv = contract.receiveFunction())
	{
		if (recv->isImplemented())
		{
			try
			{
				Json f = Json::object();
				f["name"] = "receive";
				f["params"] = Json::array();
				// Export actual parameters if the receive function has them
				for (auto const& parameter : recv->parameters())
				{
					try { f["params"].emplace_back(exportParam(*parameter)); }
					catch (...) {
						Json placeholder = Json::object();
						placeholder["name"] = parameter->name().empty() ? ("arg" + std::to_string(parameter->id())) : parameter->name();
						placeholder["type"] = Json("u256");
						f["params"].emplace_back(placeholder);
					}
				}
				// Export actual return type
				if (recv->returnParameters().empty())
					f["return"] = Json("unit");
				else if (recv->returnParameters().size() == 1)
				{
					try { f["return"] = exportTypeName(recv->returnParameters().front()->typeName()); }
					catch (...) { f["return"] = Json("u256"); }
				}
				else
					f["return"] = Json("unit");
				f["body"] = exportBody(*recv);
				f["ast_write_oracle"] = astWriteOracleJson(*recv, contract);
				functions.emplace_back(f);
			}
			catch (UnsupportedSolCore const&)
			{
				// Skip unsupported receive function
			}
			catch (std::exception const&)
			{
				// Skip receive function that causes unexpected errors during export
			}
		}
	}
	// Export fallback() function if present
	if (auto const* fb = contract.fallbackFunction())
	{
		if (fb->isImplemented())
		{
			try
			{
				Json f = Json::object();
				f["name"] = "fallback";
				f["params"] = Json::array();
				// Export actual parameters if the fallback has them
				for (auto const& parameter : fb->parameters())
				{
					try { f["params"].emplace_back(exportParam(*parameter)); }
					catch (...) {
						Json placeholder = Json::object();
						placeholder["name"] = parameter->name().empty() ? ("arg" + std::to_string(parameter->id())) : parameter->name();
						placeholder["type"] = Json("u256");
						f["params"].emplace_back(placeholder);
					}
				}
				// Export actual return type
				if (fb->returnParameters().empty())
					f["return"] = Json("unit");
				else if (fb->returnParameters().size() == 1)
				{
					try { f["return"] = exportTypeName(fb->returnParameters().front()->typeName()); }
					catch (...) { f["return"] = Json("u256"); }
				}
				else
					f["return"] = Json("unit");
				f["body"] = exportBody(*fb);
				f["ast_write_oracle"] = astWriteOracleJson(*fb, contract);
				functions.emplace_back(f);
			}
			catch (UnsupportedSolCore const&)
			{
				// Skip unsupported fallback function
			}
			catch (std::exception const&)
			{
				// Skip fallback function that causes unexpected errors during export
			}
		}
	}

	// Disambiguate overloaded function names.
	// Lean doesn't support overloading, so functions with the same name
	// get a suffix based on parameter count or parameter types.
	auto disambiguate = [](Json& funcs) {
		// Count occurrences of each name
		std::map<std::string, int> nameCounts;
		for (auto const& f : funcs)
			if (f.contains("name") && f["name"].is_string())
				nameCounts[f["name"].get<std::string>()]++;

		// For names that appear more than once, disambiguate
		std::map<std::string, int> nameIndex;
		for (auto& f : funcs)
		{
			if (!f.contains("name") || !f["name"].is_string())
				continue;
			std::string name = f["name"].get<std::string>();
			if (nameCounts[name] <= 1)
				continue;

			// Build suffix from parameter count
			int paramCount = 0;
			if (f.contains("params") && f["params"].is_array())
				paramCount = static_cast<int>(f["params"].size());

			// Build suffix from param count; if that still clashes, add type abbreviations
			std::string candidate = name + "_" + std::to_string(paramCount);
			nameIndex[candidate]++;
			if (nameIndex[candidate] > 1)
			{
				// Same name AND same param count — disambiguate with type abbreviations
				std::string typeSuffix;
				if (f.contains("params") && f["params"].is_array())
					for (auto const& p : f["params"])
					{
						if (!typeSuffix.empty()) typeSuffix += "_";
						std::string ty = "u256";
						if (p.contains("type"))
						{
							if (p["type"].is_string())
								ty = p["type"].get<std::string>();
							else if (p["type"].is_object() && p["type"].contains("kind"))
								ty = p["type"]["kind"].get<std::string>();
						}
						// Abbreviate common types
						if (ty == "address") ty = "addr";
						else if (ty == "mapping") ty = "map";
						else if (ty.length() > 4) ty = ty.substr(0, 4);
						typeSuffix += ty;
					}
				candidate = name + "_" + typeSuffix;
			}

			f["name"] = candidate;
			f["originalName"] = name;
		}
	};

	disambiguate(functions);
	disambiguate(internalFunctions);

	solcore["functions"] = std::move(functions);
	solcore["internal_functions"] = std::move(internalFunctions);

	Json events = Json::array();
	for (EventDefinition const* event: contract.events())
	{
		try
		{
			events.emplace_back(exportEvent(*event));
		}
		catch (UnsupportedSolCore const&)
		{
			// Skip events with unsupported parameter types
		}
	}
	solcore["events"] = std::move(events);

	// Export enum declarations from every source unit reachable through the
	// active compiler stack — not just the contract's own SourceUnit. Enums
	// defined in imported library files (e.g. RoundingMode in Fixed.sol) must
	// be available so downstream consumers can resolve `EnumName.VARIANT`
	// constants to their integer values. (REV-271/bug_013 fix.)
	Json enumDecls = Json::array();
	auto recordEnum = [&](EnumDefinition const& _enumDef) {
		auto const enumName = exportedEnumName(_enumDef);
		// Avoid exact duplicates while preserving distinct same-named enums
		// from different source units under qualified names.
		for (auto const& existing : enumDecls)
			if (existing.value("name", "") == enumName)
				return;
		Json decl = Json::object();
		decl["name"] = enumName;
		decl["variants"] = Json::array();
		for (auto const& member : _enumDef.members())
			decl["variants"].emplace_back(member->name());
		enumDecls.emplace_back(std::move(decl));
	};
	auto walkSourceUnit = [&](SourceUnit const& _unit) {
		forEachEnumDefinition(_unit, recordEnum);
	};
	if (activeCompilerStack)
	{
		for (auto const& sourceName : activeCompilerStack->sourceNames())
		{
			try
			{
				walkSourceUnit(activeCompilerStack->ast(sourceName));
			}
			catch (...)
			{
				// Best-effort: skip unparsable / missing units.
			}
		}
	}
	else
	{
		// Fallback for callers without a compiler stack context (legacy path).
		walkSourceUnit(contract.sourceUnit());
	}
	if (!enumDecls.empty())
		solcore["enum_decls"] = std::move(enumDecls);

	Json origins = exportOrigins(contract);
	for (auto const& [key, value]: metadata.items())
		origins[key] = value;

	return {std::move(solcore), std::move(origins)};
}

}

} // namespace

solidity::frontend::solcore::ExportArtifacts solidity::frontend::solcore::exportContract(
	CompilerStack const& _compilerStack,
	std::string const& _contractName
)
{
	try
	{
		if (VersionCompactBytes.size() >= 2 && VersionCompactBytes[0] == 0 && VersionCompactBytes[1] == 8)
			return v0_8::exportContract(_compilerStack, _contractName);
	}
	catch (UnsupportedSolCore const& exception)
	{
		return {
			unsupportedExport(_contractName, exception.what()),
			unsupportedExport(_contractName, exception.what())
		};
	}
	catch (std::exception const& exception)
	{
		// Catch any other exception that escaped (e.g., from nlohmann::json, dynamic_cast, etc.)
		std::string reason = "Internal error: "s + exception.what();
		return {
			unsupportedExport(_contractName, reason),
			unsupportedExport(_contractName, reason)
		};
	}
	catch (...)
	{
		// Catch truly unknown exceptions
		std::string reason = "Internal error: unknown exception during SolCore export";
		return {
			unsupportedExport(_contractName, reason),
			unsupportedExport(_contractName, reason)
		};
	}

	std::string reason = "No SolCore exporter is registered for compiler version " + VersionString + ".";
	return {
		unsupportedExport(_contractName, reason),
		unsupportedExport(_contractName, reason)
	};
}
