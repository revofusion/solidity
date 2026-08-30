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

#include <libsolidity/analysis/ConstantEvaluator.h>
#include <libsolidity/ast/AST.h>
#include <libsolidity/ast/ASTAnnotations.h>
#include <libsolidity/ast/TypeProvider.h>
#include <libsolidity/interface/Version.h>

#include <liblangutil/Token.h>

#include <libyul/AST.h>
#include <libyul/Dialect.h>
#include <libyul/optimiser/ASTWalker.h>
#include <libyul/Utilities.h>

#include <libsolutil/Visitor.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
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
	FunctionDefinition const* function; ///< The getter function AST node
	StructDefinition const* structDef;	///< The struct type it returns
	std::string prefix;					///< Field name prefix (e.g. "token_")
	std::string fieldName;				///< Synthetic Storage field (e.g. "token")
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
/// Exact exported name of each flattened statically-bound base target. The
/// alias is assigned from the already overload-disambiguated function name,
/// then retained so later lookups never reconstruct it from name or arity.
static thread_local std::map<FunctionDefinition const*, std::string> contractScopedSuperAliases;

/// Exported event display names, assigned once per contract export before any
/// body is exported. Declaration and emit identity is carried by the exact AST
/// declaration ID; overload disambiguation preserves stable, collision-free
/// human-readable names without participating in the join.
static thread_local std::map<EventDefinition const*, std::string> exportedEventNames;

static thread_local CompilerStack const* activeCompilerStack = nullptr;
static thread_local ContractDefinition const* activeExportContract = nullptr;
/// Stable IDs for synthesized wire names. Solidity assigns negative AST IDs to
/// compiler-generated nodes from a process-global counter, so serializing those
/// IDs makes otherwise identical exports differ across compiler invocations.
/// Pointer identity is used only as an in-process lookup key; the emitted ID is
/// assigned by deterministic export traversal and reset for every contract.
static thread_local std::map<ASTNode const*, size_t> stableSyntheticNodeIds;
static thread_local size_t nextStableSyntheticNodeId = 0;

size_t stableSyntheticNodeId(ASTNode const& _node)
{
	auto const [it, inserted] = stableSyntheticNodeIds.try_emplace(&_node, nextStableSyntheticNodeId);
	if (inserted)
		++nextStableSyntheticNodeId;
	return it->second;
}

static constexpr char EvalOrderCommitment[] = "legacy-solc-0.8/evalorder-v3";

/// Contextual target type for each byte/string literal in the function being
/// exported. Solidity's literal annotation is only `literal_string`; the
/// assignment/call/return site owns whether those same source bytes denote
/// `string`, dynamic `bytes`, or a right-padded `bytesN`.
static thread_local std::map<Literal const*, Type const*> canonicalLiteralTargets;

/// Thread-local set of exported callee names that `virtualCallTargetName`
/// resolved to a virtual slot with NO implementation anywhere in the current
/// export unit's own inheritance linearization (populated only while
/// exporting an abstract/interface unit — see `virtualCallTargetName`).
/// Cleared once per contract export, alongside `exportedFunctionNames`.
/// Recorded EXACTLY as `exportedFunctionName(*target)` would return it, so
/// the string matches byte-for-byte the name emitted into a call site's
/// `"function"` field with no re-mangling needed by consumers.
static thread_local std::set<std::string> unboundVirtualSlotNames;

// --- Internal function used as a value: bounded defunctionalization ---
//
// (SOLCORE_FNPTR_DEFUNCTIONALIZATION_DESIGN.) Solidity lets an internal
// function be passed BY VALUE into a `function(...) internal ...`-typed
// parameter (a function pointer). Every such call site used to fail closed
// at body export with "An internal function used as a value ... is not
// modeled" — AND (independently) the RECEIVING function's indirect call
// through that parameter silently name-punted to a bare `internal_call`
// using the PARAMETER's own name (see the generic FunctionCall fallback in
// exportExpr) — an accidental fail-open that would mis-bind to any real
// function sharing that name.
//
// The fix, applied only to the corpus-grounded shape (a call site passing a
// DIRECT internal-function literal — never a conditional, a function-typed
// local/storage read, or an external function value): synthesize a
// specialized sibling of the callee with the function-typed parameter(s)
// erased and every call through them resolved, at synthesis time, to the
// one statically-known target. This is bounded defunctionalization
// (Reynolds); with a statically-determined singleton candidate set per site
// it degenerates to ordinary monomorphization. Ground truth for "one
// statically-known target": solc's own IR codegen resolves an
// internal-function VALUE at the point the pointer expression is created
// (`FunctionDefinition::resolveVirtual`), with no dynamic dispatch
// afterward — specializing on that same winner is exactly the compiled
// semantics, not an approximation.
//
// Anything outside this fragment keeps failing CLOSED with a precise
// message — see `lowerInternalCalleeAndArgs` below.

/// One function-typed parameter of a specialized callee, bound to its
/// statically-resolved target.
struct FnPtrBinding
{
	VariableDeclaration const* param
		= nullptr; ///< The function-typed parameter decl, in the ORIGINAL (unspecialized) signature.
	FunctionDefinition const* target = nullptr; ///< The resolveVirtual winner bound to it; always isImplemented().
	std::string targetExportedName;				///< exportedFunctionName(*target) — virtualCallTargetName-consistent.
};

/// A specialized sibling of an internal function whose function-typed
/// parameters have been bound to static targets at one or more call sites.
struct FnPtrSpecializationRequest
{
	FunctionDefinition const* callee = nullptr; ///< The resolved implementation being specialized.
	std::string baseExportedName;				///< The name the plain (unspecialized) call would have used.
	std::vector<FnPtrBinding> bindings;			///< In original parameter order.
	std::string specializedName;
};

/// Memo of every specialization request discovered so far this export unit,
/// keyed by its final exported name — doubles as the collision-detection
/// table and the "already queued/emitted" check. Cleared once per contract
/// export, alongside `exportedFunctionNames`.
static thread_local std::map<std::string, FnPtrSpecializationRequest> fnPtrSpecializationsByName;
/// FIFO of specialization names not yet emitted into `internal_functions`.
/// Drained to a fixpoint in exportContract (transitive specialization
/// discovery re-enqueues here while draining).
static thread_local std::deque<std::string> fnPtrSpecializationQueue;
/// Non-empty ONLY while exporting the body of a specialized sibling: maps
/// each bound function-typed PARAMETER declaration to its binding, so a
/// pass-through call (`f(op)` forwarding `op` into `g(op)`) resolves `op` to
/// the already-bound target instead of failing closed.
static thread_local std::map<VariableDeclaration const*, FnPtrBinding const*> activeFnPtrBindings;
/// Per-callee memo: does this FunctionDefinition's body ever WRITE one of
/// its own function-typed parameters (Assignment LHS, `delete`, or an
/// inline-assembly external reference)? Populated once per FunctionDefinition
/// regardless of which specific targets end up bound to it — the check only
/// depends on the parameter DECLARATIONS, never on the bound targets.
static thread_local std::map<FunctionDefinition const*, bool> fnPtrCalleeAdmissibleMemo;

std::string exportedFunctionName(FunctionDefinition const& _function);
std::string exportedContractId(ContractDefinition const& _contract);
Json exportAbiDescriptor(std::string const& _name, Type const* _solidityType, bool _forLibrary);
/// One producer-proven arm of a closed internal-function candidate table.
/// The identity is structural (source unit + declaring scope + resolved
/// function signature), never reconstructed downstream from an exported name.
struct InternalFnCandidateRecord
{
	std::string identity;
	std::string tag;
	std::string functionName;
	std::string declarationContractId;
	std::string targetContractId;
	std::string delegateName;
};

/// Closed candidate set for one exact Solidity internal-function type.
/// `FunctionType::richIdentifier()` includes mutability, parameter/return
/// types, and reference locations, so it is the producer-owned type key.
struct InternalFnTableRecord
{
	std::string tableId;
	std::string dispatcherName;
	Json fnType;
	std::map<std::string, InternalFnCandidateRecord> candidatesByTag;
	std::map<std::string, std::string> tagByIdentity;
};

static thread_local std::map<std::string, InternalFnTableRecord> internalFnTablesByFingerprint;
static thread_local bool internalFnContractContainsAssembly = false;
static thread_local bool internalFnAssemblyTouchesValue = false;

/// Stable 64-bit FNV-1a tag, rendered as the exact 8-byte wire value. This is
/// deliberately independent of AST ids, traversal order, and process hash
/// randomization. Zero is reserved for an uninitialized function value.
std::string stableInternalFnTag(std::string const& _identity)
{
	unsigned long long hash = 14695981039346656037ull;
	for (char byte: _identity)
	{
		hash ^= static_cast<unsigned char>(byte);
		hash *= 1099511628211ull;
	}
	if (hash == 0)
		hash = 1;
	char const* digits = "0123456789abcdef";
	std::string result = "0x0000000000000000";
	for (size_t i = 0; i < 16; ++i)
	{
		result[17 - i] = digits[hash & 0xf];
		hash >>= 4;
	}
	return result;
}


std::string internalFnContractId(FunctionDefinition const& _function)
{
	if (auto const* contract = dynamic_cast<ContractDefinition const*>(_function.scope()))
		return exportedContractId(*contract);
	if (dynamic_cast<SourceUnit const*>(_function.scope()))
		return _function.sourceUnitName() + ":<free>";
	throw UnsupportedSolCore("Internal function value has neither a contract nor source-unit declaration scope.");
}

InternalFnTableRecord& registerInternalFnTable(FunctionType const& _fnType, Json const& _wireType)
{
	if (_fnType.kind() != FunctionType::Kind::Internal)
		throw UnsupportedSolCore("Attempted to create an internal-function table for a non-internal function type.");
	std::string fingerprint = _fnType.richIdentifier();
	std::string suffix = stableInternalFnTag("type:" + fingerprint).substr(2);
	auto insertion = internalFnTablesByFingerprint.emplace(
		fingerprint,
		InternalFnTableRecord{"internal_fn_table_" + suffix, "internal_fn_dispatch_" + suffix, _wireType, {}, {}});
	// richIdentifier is solc's equality-preserving type identity; there is
	// exactly one table per fingerprint regardless of where the type occurs.
	return insertion.first->second;
}

InternalFnCandidateRecord const& registerInternalFnCandidate(
	FunctionType const& _fnType,
	Json const& _wireType,
	FunctionDefinition const& _declaration,
	FunctionDefinition const& _target)
{
	if (!_target.isOrdinary() || !_target.isImplemented() || _target.visibility() == Visibility::External)
		throw UnsupportedSolCore(
			"Internal function value did not resolve to an implemented non-external ordinary function.");
	InternalFnTableRecord& table = registerInternalFnTable(_fnType, _wireType);
	FunctionTypePointer targetType = _target.functionType(true);
	if (!targetType)
		throw UnsupportedSolCore("Internal function candidate has no compiler-resolved internal function type.");
	std::string declarationContractId = internalFnContractId(_declaration);
	std::string targetContractId = internalFnContractId(_target);
	std::string identity = declarationContractId + "|" + _declaration.name() + "|" + targetContractId + "|"
						   + _target.name() + "|" + targetType->richIdentifier();
	std::string tag = stableInternalFnTag("candidate:" + _fnType.richIdentifier() + "|" + identity);
	auto existingIdentity = table.tagByIdentity.find(identity);
	if (existingIdentity != table.tagByIdentity.end())
		return table.candidatesByTag.at(existingIdentity->second);
	auto collision = table.candidatesByTag.find(tag);
	if (collision != table.candidatesByTag.end() && collision->second.identity != identity)
		throw UnsupportedSolCore(
			"Deterministic internal-function tag collision between '" + collision->second.identity + "' and '"
			+ identity + "'.");
	std::string functionName = exportedFunctionName(_target);
	InternalFnCandidateRecord candidate{
		identity,
		tag,
		functionName,
		declarationContractId,
		targetContractId,
		table.dispatcherName + "_delegate_" + tag.substr(2)};
	table.tagByIdentity.emplace(identity, tag);
	return table.candidatesByTag.emplace(tag, std::move(candidate)).first->second;
}

bool typeContainsInternalFnValue(Type const* _type, std::set<StructDefinition const*>& _visitingStructs)
{
	if (auto const* fnType = dynamic_cast<FunctionType const*>(_type))
		return fnType->kind() == FunctionType::Kind::Internal;
	if (auto const* arrayType = dynamic_cast<ArrayType const*>(_type))
		return typeContainsInternalFnValue(arrayType->baseType(), _visitingStructs);
	if (auto const* mappingType = dynamic_cast<MappingType const*>(_type))
		return typeContainsInternalFnValue(mappingType->valueType(), _visitingStructs);
	if (auto const* structType = dynamic_cast<StructType const*>(_type))
	{
		StructDefinition const& definition = structType->structDefinition();
		if (!_visitingStructs.insert(&definition).second)
			return false;
		bool found = false;
		for (auto const& member: definition.members())
			if (typeContainsInternalFnValue(member->annotation().type, _visitingStructs))
			{
				found = true;
				break;
			}
		_visitingStructs.erase(&definition);
		return found;
	}
	return false;
}

bool typeContainsInternalFnValue(Type const* _type)
{
	std::set<StructDefinition const*> visitingStructs;
	return typeContainsInternalFnValue(_type, visitingStructs);
}

bool contractHasStoredInternalFnValue(ContractDefinition const& _contract)
{
	for (ContractDefinition const* source: _contract.annotation().linearizedBaseContracts)
		for (VariableDeclaration const* stateVar: source->stateVariables())
			if (!stateVar->isConstant() && !stateVar->immutable()
				&& typeContainsInternalFnValue(stateVar->annotation().type))
				return true;
	return false;
}

bool contractHasAggregateInternalFnValue(ContractDefinition const& _contract)
{
	struct Detector: ASTConstVisitor
	{
		bool found = false;

		bool visit(VariableDeclaration const& _declaration) override
		{
			Type const* type = _declaration.annotation().type;
			auto const* directFnType = dynamic_cast<FunctionType const*>(type);
			bool const directInternalFn
				= directFnType && directFnType->kind() == FunctionType::Kind::Internal;
			if (!directInternalFn && typeContainsInternalFnValue(type))
				found = true;
			return !found;
		}
	};

	for (ContractDefinition const* source: _contract.annotation().linearizedBaseContracts)
	{
		Detector detector;
		source->accept(detector);
		if (detector.found)
			return true;
	}
	return false;
}

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
	explicit UncheckedBlockGuard(bool _unchecked): m_previous(inUncheckedBlock) { inUncheckedBlock = _unchecked; }
	~UncheckedBlockGuard() { inUncheckedBlock = m_previous; }
	UncheckedBlockGuard(UncheckedBlockGuard const&) = delete;
	UncheckedBlockGuard& operator=(UncheckedBlockGuard const&) = delete;

private:
	bool m_previous;
};

/// Record checkedness explicitly on every arithmetic node whose semantics
/// differ between checked and unchecked Solidity contexts. Consumers reject
/// missing/null metadata rather than guessing checked execution.
void markUncheckedContext(Json& _result)
{
	_result["unchecked"] = inUncheckedBlock;
}

// --- Side-effect hoisting for mutating sub-expressions ---
//
// `x++` / `x--` / `++x` / `--x` and assignments are EXPRESSIONS in Solidity,
// so they can appear nested arbitrarily deep inside a larger expression
// (e.g. `abi.encode(nonces[owner]++, ...)` in the classic permit pattern).
// exportExpr used to lower such a nested mutation to just its pure value —
// silently DROPPING the underlying storage/local write, which made the
// generated model unsound (a `preserves_storage_except` claim could hold in
// the model while the real contract writes the field). Postfix `x++` was
// additionally lowered to the WRONG value (`x+1` instead of the pre-value).
//
// The fix: every statement export installs a HoistScope. When exportExpr
// meets a mutating sub-expression in a hoist-safe position, it appends an
// explicit capture-let plus the real mutation statement to the scope's
// buffer and substitutes a read of the temporary at the original use site.
// exportStmt then emits `block { <hoisted...>, <statement> }` (the OCaml
// frontend flattens nested blocks into the surrounding statement list, so
// the hoisted statements execute immediately before the statement and any
// `let` inside the wrapped statement stays visible to later statements).
//
// Soundness of moving the write to just before the statement: Solidity
// deliberately leaves intra-expression evaluation order unspecified (only
// statement order and short-circuiting are guaranteed), and solc's two
// codegen pipelines actually differ here, so the model must only commit to
// an order when every order is observationally equivalent. That is enforced
// by checkHoistConflictsOrThrow: the mutated base variable must not be
// referenced anywhere else in the statement's once-evaluated header
// expressions, and (for storage-backed targets) every call reachable from
// those expressions must be a storage-inert builtin or an internal
// view/pure function that provably never touches the mutated variable.
// Anything else fails CLOSED (UnsupportedSolCore -> unsupported_body), it
// is never silently dropped again. Known accepted divergence: when the
// hoisted checked arithmetic AND another sub-expression of the same
// statement would both revert, the model may report the other revert
// reason than the compiled code (the revert SET is identical either way;
// only the reason payload of such double-revert executions can differ).
//
// Repeated positions (loop conditions) still cannot be hoisted. Guarded
// one-shot positions (ternary arms and `&&`/`||` right operands) are lowered
// structurally into a branch-local scope below, so their effects remain under
// the selecting guard.
struct HoistScope
{
	/// Hoisted statements, in required execution order.
	std::vector<Json> statements;
	/// The enclosing statement's once-evaluated header expressions (the
	/// conflict-scan domain; empty means hoisting is not allowed here).
	std::vector<Expression const*> roots;
	/// AST node id -> temp local name. Some export paths legitimately
	/// export the same sub-expression more than once (compound-assignment
	/// lvalues, exception-based lowering fallbacks); the memo makes the
	/// second export reuse the already-hoisted temp instead of duplicating
	/// the side effect.
	std::map<int64_t, std::string> memo;
	/// Storage-path expression ids whose key/index snapshot lets have already
	/// been committed to this statement. Some assignment paths export the same
	/// rooted expression more than once; the captured keys must still execute
	/// exactly once.
	std::set<int64_t> storageRefSnapshotOwners;
	bool allowed = false;
};

static thread_local HoistScope* activeHoistScope = nullptr;

/// RAII: installs a fresh HoistScope for one statement export.
struct HoistScopeGuard
{
	explicit HoistScopeGuard(std::vector<Expression const*> _roots): m_previous(activeHoistScope)
	{
		m_scope.roots = std::move(_roots);
		m_scope.allowed = !m_scope.roots.empty();
		activeHoistScope = &m_scope;
	}
	~HoistScopeGuard() { activeHoistScope = m_previous; }
	HoistScopeGuard(HoistScopeGuard const&) = delete;
	HoistScopeGuard& operator=(HoistScopeGuard const&) = delete;

	/// Wrap the exported statement with any hoisted statements.
	Json wrap(Json&& _stmt)
	{
		if (m_scope.statements.empty())
			return std::move(_stmt);
		Json block = Json::object();
		block["kind"] = "block";
		block["statements"] = Json::array();
		for (auto& hoisted: m_scope.statements)
			block["statements"].emplace_back(std::move(hoisted));
		block["statements"].emplace_back(std::move(_stmt));
		m_scope.statements.clear();
		return block;
	}
	/// Move the statements produced inside this scope to a structurally
	/// selected parent branch.
	std::vector<Json> takeStatements() { return std::move(m_scope.statements); }

private:
	HoistScope m_scope;
	HoistScope* m_previous;
};


// --- Declared-width signal for unsigned arithmetic and left shifts ---
//
// Width is part of the operation. Every emitted
// u256_add/u256_sub/u256_mul/u256_exp/u256_shl node carries the exact
// type-checker-resolved width, including an explicit 256 for a full word.
// Fixed bytes participate only in left shifts and use their exact lane width
// on SolCore's right-aligned carrier.
void tagUnsignedArithWidth(Json& _result, Type const* _operationType)
{
	if (auto const* intType = dynamic_cast<IntegerType const*>(_operationType))
	{
		if (intType->isSigned())
			throw UnsupportedSolCore("Unsigned arithmetic node received a signed operation type.");
		_result["bits"] = static_cast<int>(intType->numBits());
		return;
	}
	if (auto const* fixedBytes = dynamic_cast<FixedBytesType const*>(_operationType))
	{
		_result["bits"] = static_cast<int>(8 * fixedBytes->numBytes());
		return;
	}
	if (
		_operationType
		&& _operationType->category() == Type::Category::RationalNumber
	)
	{
		// Untyped integer constant expressions are exact rationals in solc's
		// AST. Their SolCore carrier is the full word; any contextual narrow
		// conversion is emitted by the enclosing conversion node.
		_result["bits"] = 256;
		return;
	}
	throw UnsupportedSolCore(
		"Width-dependent unsigned arithmetic/shift lacks an exact producer-resolved integer width.");
}

// --- Signed declared-width signal for the signed-integer primitives family
// (SolCore design: "Signed integer comparison, shift, and division
// primitives") ---
//
// Every SolCore value of declared type `intN` is represented at the value
// level as the full 256-bit sign-extended two's-complement word
// (`fromSigned` of its mathematical value). Comparisons and `>>` (SAR) are
// width-independent under this representation (one Lean op serves
// int8..int256), but division/modulo/checked-or-wrapping add/sub/mul/neg
// must know the DECLARED width to bound/overflow-check/wrap correctly --
// exactly the same "operation type is the only sound width source" argument
// `tagUnsignedArithWidth` documents above.
// `signedOperationBits` returns the operand width whenever `_type` is a
// signed `IntegerType`, so callers can select the signed lowering and tag
// its `"bits"` field; std::nullopt means "not a signed integer operation"
// (caller keeps the unsigned/status-quo lowering).
std::optional<unsigned> signedOperationBits(Type const* _type)
{
	if (!_type || _type->category() != Type::Category::Integer)
		return std::nullopt;
	auto const* intType = dynamic_cast<IntegerType const*>(_type);
	if (intType && intType->isSigned())
		return intType->numBits();
	return std::nullopt;
}

std::string exportedFunctionName(FunctionDefinition const& _function);

template<class F>
void forEachEnumDefinition(SourceUnit const& _unit, F&& _f)
{
	for (auto const& node: _unit.nodes())
	{
		if (auto const* enumDef = dynamic_cast<EnumDefinition const*>(node.get()))
			_f(*enumDef);
		else if (auto const* contractNode = dynamic_cast<ContractDefinition const*>(node.get()))
			for (auto const& subNode: contractNode->subNodes())
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
	// The scan below is a pure function of the compiler stack: O(sources x
	// nodes) with a dynamic_cast per node. Callers ask once per exported enum
	// NAME OCCURRENCE (per contract, per field/param/event), so scanning
	// per call is O(occurrences x sources x nodes) — the dominant export
	// cost on multi-source inputs. Build the name → identities index once
	// per stack and answer by lookup; the predicate is unchanged
	// (qualify iff some OTHER identity shares the bare name).
	static thread_local CompilerStack const* cachedEnumStack = nullptr;
	static thread_local std::map<std::string, std::set<std::string>> enumIdentitiesByName;
	if (cachedEnumStack != activeCompilerStack)
	{
		enumIdentitiesByName.clear();
		for (auto const& sourceName: activeCompilerStack->sourceNames())
		{
			try
			{
				forEachEnumDefinition(
					activeCompilerStack->ast(sourceName),
					[&](EnumDefinition const& candidate)
					{
						enumIdentitiesByName[candidate.name()].insert(enumDefinitionIdentity(candidate));
					});
			}
			catch (...)
			{
			}
		}
		cachedEnumStack = activeCompilerStack;
	}
	auto const it = enumIdentitiesByName.find(_enumDef.name());
	if (it == enumIdentitiesByName.end())
		return false;
	auto const ownIdentity = enumDefinitionIdentity(_enumDef);
	for (auto const& identity: it->second)
		if (identity != ownIdentity)
			return true;
	return false;
}

std::string exportedEnumName(EnumDefinition const& _enumDef)
{
	if (enumNameNeedsQualification(_enumDef))
		return enumDefinitionIdentity(_enumDef);
	return _enumDef.name();
}

template<class F>
void forEachStructDefinition(SourceUnit const& _unit, F&& _f)
{
	for (auto const& node: _unit.nodes())
	{
		if (auto const* structDef = dynamic_cast<StructDefinition const*>(node.get()))
			_f(*structDef);
		else if (auto const* contractNode = dynamic_cast<ContractDefinition const*>(node.get()))
			for (auto const* structDef: contractNode->definedStructs())
				_f(*structDef);
	}
}

std::string sanitizedIdentifierComponent(std::string const& _raw)
{
	std::string out;
	for (char c: _raw)
		out += (std::isalnum(static_cast<unsigned char>(c)) || c == '_') ? c : '_';
	if (out.empty() || std::isdigit(static_cast<unsigned char>(out[0])))
		out = "_" + out;
	return out;
}

/// Scope-qualified struct spelling used when the bare name collides:
/// `<ScopeContract>_<Name>` for contract/library/interface-scoped structs,
/// `<SourceStem>_<Name>` for file-level structs. The suffix `_<astId>` is
/// appended only when qualification itself still collides.
std::string qualifiedStructName(StructDefinition const& _structDef)
{
	if (auto const* scope = dynamic_cast<ContractDefinition const*>(_structDef.scope()))
		return sanitizedIdentifierComponent(scope->name()) + "_" + _structDef.name();
	std::string stem = _structDef.sourceUnitName();
	if (auto const slash = stem.find_last_of('/'); slash != std::string::npos)
		stem = stem.substr(slash + 1);
	if (auto const dot = stem.find('.'); dot != std::string::npos)
		stem = stem.substr(0, dot);
	return sanitizedIdentifierComponent(stem) + "_" + _structDef.name();
}

/// Two distinct struct DEFINITIONS sharing one bare name would collapse to a
/// single exported record, silently binding every user of the suppressed
/// declaration to the survivor's layout (e.g. PoolOracle.Observation vs
/// IPoolTape.Observation). Mirror [exportedEnumName]: qualify iff some OTHER
/// definition shares the bare name anywhere in the compiler stack.
std::string exportedStructName(StructDefinition const& _structDef)
{
	if (!activeCompilerStack)
		return _structDef.name();
	static thread_local CompilerStack const* cachedStructStack = nullptr;
	static thread_local std::map<std::string, std::set<StructDefinition const*>> structDefsByName;
	static thread_local std::map<std::string, std::set<StructDefinition const*>> structDefsByQualifiedName;
	if (cachedStructStack != activeCompilerStack)
	{
		structDefsByName.clear();
		structDefsByQualifiedName.clear();
		for (auto const& sourceName: activeCompilerStack->sourceNames())
		{
			try
			{
				forEachStructDefinition(
					activeCompilerStack->ast(sourceName),
					[&](StructDefinition const& candidate)
					{
						structDefsByName[candidate.name()].insert(&candidate);
						structDefsByQualifiedName[qualifiedStructName(candidate)].insert(&candidate);
					});
			}
			catch (...)
			{
			}
		}
		cachedStructStack = activeCompilerStack;
	}
	auto const it = structDefsByName.find(_structDef.name());
	if (it == structDefsByName.end() || it->second.size() <= 1)
		return _structDef.name();
	std::string qualified = qualifiedStructName(_structDef);
	auto const qit = structDefsByQualifiedName.find(qualified);
	if (qit != structDefsByQualifiedName.end() && qit->second.size() > 1)
		qualified += "_" + std::to_string(_structDef.id());
	return qualified;
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
	auto const* funcDef = dynamic_cast<FunctionDefinition const*>(_callee.annotation().referencedDeclaration);
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
bool isNamespacedStorageGetter(FunctionDefinition const& _function, StructDefinition const** _outStructDef)
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

// --- General local storage-reference-variable alias tracking (Phase 1a) ---
//
// Generalizes the ERC-7201 namespaced-alias mechanism above (which only
// handles `X storage $ = _getXStorage();`, keyed by NAME, prefix-only, no
// dynamic keys) to the general Solidity pattern `T storage x = <storage
// lvalue>;` where `<storage lvalue>` may thread through mapping/array
// indices and struct-field chains. See documentation/... design doc
// "General Local Storage-Reference-Variable Alias Tracking" §3.
//
// Keyed by VariableDeclaration* (not name) to avoid the shadowing hazard the
// legacy namespacedStorageAliases string map carries; that map's own
// behavior is left completely untouched by this section.
struct StorageRefStep
{
	enum class Kind
	{
		Field,
		MappingKey,
		ArrayIndex
	} kind;
	std::string field;	 ///< Kind::Field: struct member / flattened field name.
	std::string keyTemp; ///< Kind::MappingKey / ArrayIndex: snapshot temp local name.
};

struct StorageRefTarget
{
	enum class RootKind
	{
		StateField,
		LocalParameter,
		RawSlot
	};

	std::string root;				   ///< State field or threaded storage parameter name.
	std::vector<StorageRefStep> steps; ///< Access steps in order, outermost first.
	RootKind rootKind = RootKind::StateField;
	Type const* rootType = nullptr; ///< Exact typed root for residual structural references.
	Json rawSlot; ///< RootKind::RawSlot: exact caller-context slot-word expression.
	bool snapshotAsTypedCall = false; ///< Raw-slot reinterpretation: pin the pure helper call, not a differently typed root alias.

	StorageRefTarget() = default;
	StorageRefTarget(
		std::string _root,
		std::vector<StorageRefStep> _steps,
		RootKind _rootKind = RootKind::StateField,
		Type const* _rootType = nullptr,
		Json _rawSlot = Json{}):
		root(std::move(_root)),
		steps(std::move(_steps)),
		rootKind(_rootKind),
		rootType(_rootType),
		rawSlot(std::move(_rawSlot))
	{}
};

bool storageRefTargetsEqual(StorageRefTarget const& _lhs, StorageRefTarget const& _rhs)
{
	if (_lhs.root != _rhs.root || _lhs.rootKind != _rhs.rootKind || _lhs.steps.size() != _rhs.steps.size()
		|| _lhs.snapshotAsTypedCall != _rhs.snapshotAsTypedCall
		|| (_lhs.rootKind == StorageRefTarget::RootKind::RawSlot && _lhs.rawSlot != _rhs.rawSlot))
		return false;
	for (size_t i = 0; i < _lhs.steps.size(); ++i)
	{
		StorageRefStep const& lhsStep = _lhs.steps[i];
		StorageRefStep const& rhsStep = _rhs.steps[i];
		if (lhsStep.kind != rhsStep.kind || lhsStep.field != rhsStep.field || lhsStep.keyTemp != rhsStep.keyTemp)
			return false;
	}
	return true;
}

/// Tracked, RESOLVED storage-ref aliases for the function currently being
/// exported. Presence in this map means "safe to substitute at every use
/// site in this function" — see the deviation note below for why this
/// implementation does not additionally need a POISONED tri-state.
///
/// Deviation from the design doc's §3.1/§3.4: the design specifies a
/// flow-sensitive analysis with an `optional<StorageRefTarget>` value
/// (nullopt = "poisoned": bound then invalidated by a conditional/loop
/// rebind) plus per-branch join rules and a loop pre-scan. This
/// implementation instead performs one whole-function conservative
/// pre-pass (StorageRefRebindScanner, below) that finds every storage-
/// located local ever reassigned via a bare-identifier Assignment ANYWHERE
/// in the function (straight-line, in a branch, or in a loop) and simply
/// never registers such a variable as trackable at all. This is strictly
/// MORE conservative than the design's flow-sensitive join/poison rules
/// (it refuses tracking for some straight-line-safe rebind patterns the
/// fuller design would accept), but it is sound by the same argument
/// (§3.6's fallback policy: an untracked bind falls through to today's
/// status-quo copy-`let` lowering, which is unchanged behavior) and is
/// much simpler to implement and audit correctly. None of the concrete
/// cited instances in the design rely on tracking through a rebind, so
/// this narrowing does not affect the capability's headline cases.
static thread_local std::map<VariableDeclaration const*, StorageRefTarget> storageRefAliasTargets;

/// Whole-function pre-pass result: storage-located locals that are REBOUND
/// (reassigned via a bare-identifier Assignment) somewhere in the function
/// and must therefore never be registered in storageRefAliasTargets (see
/// the deviation note above).
static thread_local std::set<VariableDeclaration const*> storageRefNeverTrack;
/// Locals and parameters represented as first-class structural StorageRef
/// values. This set is deliberately disjoint from storageRefAliasTargets:
/// static aliases remain substitution-only and never acquire a runtime value.
static thread_local std::set<VariableDeclaration const*> storageRefValueLocals;

/// True while exporting a function whose single storage-reference return is a
/// residual value (currently conditional library returns). Return statements
/// must transport the reference itself rather than dereference it.
static thread_local bool activeResidualStorageRefReturn = false;

/// Public-library ABI entry parameters arrive as caller-relative raw slot
/// words. exportFunction installs this set only while exporting that primary
/// entry; exportBody shadows each word with a typed storage_ref_raw_slot.
static thread_local std::set<VariableDeclaration const*> activeRawSlotStorageRefParams;

/// [P0 fail-closed: E0(a') — no silently dropped writes THROUGH an untracked
/// storage-pointer local] Whole-function pre-pass result: every storage-located
/// declaration that some statement WRITES THROUGH (`p.f = v`, `p[i] = v`,
/// `delete p`, `p.push(..)`/`p.pop()`, `p[i]++`), as opposed to merely rebinding
/// the pointer itself (`p = q`). Filled by StorageRefWriteThroughScanner (below,
/// next to StorageRefShrinkScanner) and consumed at the copy-`let` fallback in
/// the variable-declaration lowering.
static thread_local std::set<VariableDeclaration const*> storageRefWriteThroughLocals;

/// [E2(b) — single-assignment storage-pointer binding] Whole-function
/// pre-pass result: storage-located locals declared WITHOUT an initializer
/// and assigned exactly ONCE, by a bare statement-position `p = <rhs>;`.
/// solc's definite-assignment rule guarantees every use of such a pointer is
/// dominated by that assignment, so the assignment is treated as the binding
/// initializer: the assignment statement resolves its RHS through
/// `resolveStorageRefInitializer` and registers the alias (see
/// exportAssignment). The declaration-site E0(a') write-through guard defers
/// to the assignment site for these locals — which must therefore refuse
/// (never fall back silently) when the RHS does not resolve and the local is
/// written through. Filled from StorageRefRebindScanner in exportBody.
static thread_local std::set<VariableDeclaration const*> storageRefSingleAssignBindable;

/// [P0 fail-closed: storage-ref early binding vs. array shrink] Whole-function
/// pre-pass result: the state-variable roots whose storage arrays this
/// function can SHRINK (`pop()`, `delete`, whole reassignment), plus a
/// wildcard flag for a shrink whose root could not be resolved to a state
/// variable. Filled by StorageRefShrinkScanner (below, next to
/// StorageRefTarget) and consumed at storage-ref bind sites.
struct StorageRefShrinkInfo
{
	std::set<std::string> shrunkRoots;
	bool anyUnresolvedShrink = false;
};

static thread_local StorageRefShrinkInfo storageRefShrinkInfo;

/// RAII: save/clear/restore storageRefAliasTargets and storageRefNeverTrack
/// around one function's export, mirroring NamespacedStorageScope's idiom.
struct StorageRefAliasScope
{
	std::map<VariableDeclaration const*, StorageRefTarget> savedTargets;
	std::set<VariableDeclaration const*> savedNeverTrack;
	std::set<VariableDeclaration const*> savedValueLocals;
	bool savedResidualReturn = false;
	std::set<VariableDeclaration const*> savedWriteThrough;
	std::set<VariableDeclaration const*> savedSingleAssignBindable;
	StorageRefShrinkInfo savedShrinkInfo;
	StorageRefAliasScope()
	{
		savedTargets = storageRefAliasTargets;
		savedNeverTrack = storageRefNeverTrack;
		savedValueLocals = storageRefValueLocals;
		savedResidualReturn = activeResidualStorageRefReturn;
		savedWriteThrough = storageRefWriteThroughLocals;
		savedSingleAssignBindable = storageRefSingleAssignBindable;
		savedShrinkInfo = storageRefShrinkInfo;
		storageRefAliasTargets.clear();
		storageRefNeverTrack.clear();
		storageRefValueLocals.clear();
		activeResidualStorageRefReturn = false;
		storageRefWriteThroughLocals.clear();
		storageRefSingleAssignBindable.clear();
		storageRefShrinkInfo = StorageRefShrinkInfo{};
	}
	~StorageRefAliasScope()
	{
		storageRefAliasTargets = savedTargets;
		storageRefValueLocals = savedValueLocals;
		activeResidualStorageRefReturn = savedResidualReturn;
		storageRefNeverTrack = savedNeverTrack;
		storageRefWriteThroughLocals = savedWriteThrough;
		storageRefSingleAssignBindable = savedSingleAssignBindable;
		storageRefShrinkInfo = savedShrinkInfo;
	}
};

/// Pre-pass: find every storage-located local rebound via a bare-identifier
/// Assignment anywhere in a function body. Run once per function BEFORE
/// exportStmt walks it, so bind-site resolution (below) can consult the
/// result and simply skip tracking for such variables.
struct StorageRefRebindScanner: ASTConstVisitor
{
	std::set<VariableDeclaration const*> neverTrack;
	/// [E2(b)] Bookkeeping for the single-assignment binding rule (see
	/// storageRefSingleAssignBindable below): how many times each storage-ref
	/// local is assigned ANYWHERE (any position), how many of those are the
	/// rewriteable bare statement-position shape (`p = <path>;` as its own
	/// ExpressionStatement), which locals are rebound through a
	/// tuple-destructuring component (never bindable), and which are declared
	/// WITH an initializer (the ordinary bind-at-declaration path owns those).
	std::map<VariableDeclaration const*, size_t> assignCounts;
	std::map<VariableDeclaration const*, size_t> statementAssignCounts;
	std::set<VariableDeclaration const*> tupleRebound;
	std::set<VariableDeclaration const*> declaredWithInitializer;

	static VariableDeclaration const* storageRefDeclOf(Expression const& _target)
	{
		if (auto const* ident = dynamic_cast<Identifier const*>(&_target))
			if (auto const* decl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
				if (!decl->isStateVariable() && decl->referenceLocation() == VariableDeclaration::Location::Storage)
					return decl;
		return nullptr;
	}

	void noteIfStorageRefRebind(Expression const& _target, bool _inTuple)
	{
		if (auto const* decl = storageRefDeclOf(_target))
		{
			neverTrack.insert(decl);
			assignCounts[decl] += 1;
			if (_inTuple)
				tupleRebound.insert(decl);
		}
	}

	bool visit(ExpressionStatement const& _stmt) override
	{
		// Statement-position bare rebind `p = <rhs>;` — the only assignment
		// shape the [E2(b)] bind-at-assignment lowering can rewrite (an
		// expression-position assignment has a value consumer and no
		// statement slot for the key-snapshot lets).
		if (auto const* assignment = dynamic_cast<Assignment const*>(&_stmt.expression()))
			if (assignment->assignmentOperator() == Token::Assign)
				if (auto const* decl = storageRefDeclOf(assignment->leftHandSide()))
					statementAssignCounts[decl] += 1;
		return true;
	}

	bool visit(VariableDeclarationStatement const& _stmt) override
	{
		if (_stmt.initialValue())
			for (auto const& declared: _stmt.declarations())
				if (declared && !declared->isStateVariable()
					&& declared->referenceLocation() == VariableDeclaration::Location::Storage)
					declaredWithInitializer.insert(declared.get());
		return true;
	}

	bool visit(Assignment const& _assignment) override
	{
		// A bare-identifier LHS is the common case; a storage-ref local can
		// ALSO be rebound as one component of a tuple-destructuring
		// assignment (`(x, y) = (a[i], b[j]);`) — walk every component so
		// that shape is not missed (missing it here would be a soundness
		// gap: the pre-rebind target would keep being substituted at uses
		// after the rebind).
		if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_assignment.leftHandSide()))
		{
			for (auto const& component: tuple->components())
				if (component)
					noteIfStorageRefRebind(*component, /*_inTuple=*/true);
		}
		else
			noteIfStorageRefRebind(_assignment.leftHandSide(), /*_inTuple=*/false);
		return true;
	}

	/// [E2(b)] Locals admitted to the single-assignment binding rule:
	/// declared WITHOUT an initializer and assigned exactly once, by a bare
	/// statement-position `p = <rhs>;`. solc's own definite-assignment check
	/// makes any use before that assignment a compile error, so the single
	/// assignment dominates every use — the flow-insensitive, provably
	/// sufficient condition of design E2(b). Everything else keeps today's
	/// never-track treatment.
	std::set<VariableDeclaration const*> singleAssignBindable() const
	{
		std::set<VariableDeclaration const*> bindable;
		for (auto const& [decl, count]: assignCounts)
		{
			auto stmtIt = statementAssignCounts.find(decl);
			if (count == 1 && stmtIt != statementAssignCounts.end() && stmtIt->second == 1 && !tupleRebound.count(decl)
				&& !declaredWithInitializer.count(decl))
				bindable.insert(decl);
		}
		return bindable;
	}
};

std::string storageRefKeyTempName(ASTNode const& _owner, size_t _idx)
{
	return "__solcore_sref_" + std::to_string(stableSyntheticNodeId(_owner)) + "_k" + std::to_string(_idx);
}

Json localExpr(std::string const& _name);

/// True only for a declared input parameter whose storage-reference value is
/// already modeled by Base.ml's copy-in/copy-out threading. A storage local is
/// deliberately excluded: unless it is present in storageRefAliasTargets it is
/// still only a value copy and accepting it would drop the write.
bool isStorageRefParameter(VariableDeclaration const* _decl)
{
	return _decl && !_decl->isStateVariable() && _decl->referenceLocation() == VariableDeclaration::Location::Storage
		   && _decl->isCallableOrCatchParameter() && !_decl->isReturnParameter();
}

/// Build the read-position JSON for a resolved alias target: reconstructs
/// the storage_get/storage_map_get root plus every captured step, exactly
/// matching the canonical inline nested-storage-read shape the exporter
/// already produces for a direct (non-aliased) chain (design §3.3/§7.1).
Json aliasReadJson(StorageRefTarget const& _target)
{
	Json current = Json::object();
	size_t i = 0;
	if (_target.rootKind == StorageRefTarget::RootKind::LocalParameter)
		current = localExpr(_target.root);
	else if (!_target.steps.empty() && _target.steps.front().kind == StorageRefStep::Kind::MappingKey)
	{
		current["kind"] = "storage_map_get";
		current["field"] = _target.root;
		current["key"] = localExpr(_target.steps.front().keyTemp);
		i = 1;
	}
	else
	{
		current["kind"] = "storage_get";
		current["field"] = _target.root;
	}
	for (; i < _target.steps.size(); ++i)
	{
		StorageRefStep const& step = _target.steps[i];
		Json next = Json::object();
		if (step.kind == StorageRefStep::Kind::Field)
		{
			next["kind"] = "field";
			next["base"] = current;
			next["field"] = step.field;
		}
		else if (step.kind == StorageRefStep::Kind::MappingKey)
		{
			next["kind"] = "array_get";
			next["base"] = current;
			next["index"] = localExpr(step.keyTemp);
		}
		else
		{
			next["kind"] = "internal_call";
			next["function"] = "storage_array_slot_get";
			next["args"] = Json::array();
			next["args"].emplace_back(std::move(current));
			next["args"].emplace_back(localExpr(step.keyTemp));
		}
		current = next;
	}
	return current;
}

/// True iff `_decl` is a tracked, resolved storage-ref alias in the
/// function currently being exported.
bool isTrackedStorageRefAlias(VariableDeclaration const* _decl)
{
	return _decl && storageRefAliasTargets.count(_decl) > 0;
}

enum class StorageRefKeySnapshotMode
{
	StableOnly,
	DirectMappingKey,
	OrderedPath
};

std::optional<Json> exportResolvedStorageRefUse(
	Expression const& _expr, StorageRefKeySnapshotMode _snapshotMode = StorageRefKeySnapshotMode::StableOnly);


/// Peel an assignment lvalue down to the expression its access chain is
/// ROOTED at: `L3.whole(s0).m[k]` -> the `L3.whole(s0)` FunctionCall,
/// `arr[i].f` -> the `arr` Identifier. Single-component parenthesised
/// expressions are transparent so `(p).x` peels exactly like `p.x`.
Expression const* peelLValueRoot(Expression const& _lhs)
{
	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_lhs))
		return peelLValueRoot(memberAccess->expression());
	if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_lhs))
		return peelLValueRoot(indexAccess->baseExpression());
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_lhs))
		if (!tuple->isInlineArray() && tuple->components().size() == 1 && tuple->components().front())
			return peelLValueRoot(*tuple->components().front());
	return &_lhs;
}

/// Recognize an assignment lvalue rooted at Solidity's typed, zero-argument
/// storage-array `push()`. The call returns the newly appended storage slot;
/// treating it as an ordinary value duplicates the push during lvalue
/// validation and then discards the field update.
FunctionCall const* storageArrayPushLValueRoot(Expression const& _lhs)
{
	auto const* call = dynamic_cast<FunctionCall const*>(peelLValueRoot(_lhs));
	if (!call || !call->arguments().empty())
		return nullptr;
	auto const* member = dynamic_cast<MemberAccess const*>(&call->expression());
	if (!member || member->memberName() != "push")
		return nullptr;
	auto const* functionType = dynamic_cast<FunctionType const*>(member->annotation().type);
	auto const* arrayType = dynamic_cast<ArrayType const*>(member->expression().annotation().type);
	if (!functionType || functionType->kind() != FunctionType::Kind::ArrayPush || !arrayType
		|| arrayType->location() != DataLocation::Storage || !arrayType->isDynamicallySized())
		return nullptr;
	return call;
}

/// [P0 fail-closed: no silently dropped storage writes]
///
/// Several assignment lowerings below emit a DISCARDED `expr` statement
/// (`array_set_expr` / `struct_update` / `generic_assign` / `tuple_assign`)
/// rather than a real `storage_set`/`storage_map_set`/`array_set`. Those are
/// only sound because the generator's writeback pass re-roots them into a
/// genuine read-modify-write storage update — and it can only do that when
/// the exported base expression bottoms out in a recognised storage ROOT
/// (`storage_get` / `storage_map_get`, i.e. a state variable or a tracked
/// storage-ref alias).
///
/// A statically resolvable call-returned storage path is substituted through
/// [exportResolvedStorageRefUse] before the discarded functional update is
/// built, so the frontend writeback walker sees the same StorageGet /
/// StorageMapGet-rooted chain as a direct lvalue. Conditional, multi-return,
/// effectful, and otherwise unresolved call roots still arrive here and keep
/// the established refusal below. Identifier roots are deliberately left
/// alone: state variables, tracked aliases, and storage parameters have
/// dedicated exact models, while an untracked storage local is caught by the
/// write-through oracle. Memory/calldata roots remain ordinary value updates.
void requireLoweredStorageWriteRootOrThrow(Expression const& _lhs, std::string const& _shape)
{
	Expression const* root = peelLValueRoot(_lhs);
	if (dynamic_cast<Identifier const*>(root))
		return;
	Type const* rootType = root->annotation().type;
	if (!rootType || !rootType->dataStoredIn(DataLocation::Storage))
		return;
	// The ERC-7201 namespaced-storage getter (`_getXStorage().field…`) is a
	// RECOGNISED call root even though it is a FunctionCall: exportExpr
	// lowers it to a rooted `storage_get` carrying the flattened namespace
	// prefix (see namespacedGetterPrefix), so the writeback pass re-roots
	// these writes exactly like a bare state-variable access. Not a drop.
	if (auto const* call = dynamic_cast<FunctionCall const*>(root))
	{
		if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
			if (namespacedGetterPrefix(*callee))
				return;
		if (exportResolvedStorageRefUse(*call))
			return;
	}
	throw UnsupportedSolCore(
		"Assignment through a storage pointer that is not rooted at a state variable or a "
		"tracked storage-ref alias is not modeled ("
		+ _shape
		+ "): the write would be "
		  "exported as a discarded expression statement that the storage writeback pass cannot "
		  "re-root, silently dropping the state mutation. Refusing instead of modeling a "
		  "function that leaves storage unchanged.");
}

/// [P0 fail-closed: E0(a') — writes THROUGH an untracked storage-pointer local]
///
/// Whole-function pre-pass, same conservative spirit as StorageRefRebindScanner
/// and StorageRefShrinkScanner: collect every storage-located declaration that
/// a statement WRITES THROUGH.
///
/// Why it matters (E-BUG-3, re-derived 2026-07-25 in the emitted Lean for
/// `storage-ref-path-return` t1/t3/t5/t6/t7): when a storage-pointer local's
/// initializer cannot be resolved to a storage PATH (today: every call
/// initializer — `resolveStorageRefInitializer` returns nullopt for
/// `FunctionCall` roots — plus every never-tracked/rebound local), the
/// declaration falls through to the status-quo copy-`let` lowering, which binds
/// a VALUE copy of the pointed-to data. A later write through that local is
/// then exported as either a bare-identifier `assign` (`s.a = 42` becomes
/// `assign s := struct_update(local s, a, 42)`) or a discarded `expr`
/// (`p[0].a = 99`), and BOTH lower to a dead local rebind in Lean: storage is
/// never updated and the function returns the untouched field. The real EVM
/// writes through the pointer. `requireLoweredStorageWriteRootOrThrow` above
/// deliberately exempts Identifier roots, and the generator's
/// `demote_discarded_storage_write_functions` backstop only fires for
/// storage-ROOTED discarded expressions, so neither guard sees this shape —
/// it scored fully green while dropping the write.
///
/// REBINDS ARE NOT WRITE-THROUGHS. `p = q` (a bare-identifier Assignment with
/// no access steps) re-points the pointer; under the copy-`let` model it
/// becomes a fresh value copy, which is value-correct at that point and is
/// already handled by StorageRefRebindScanner (never-track). Only lvalues with
/// at least one member/index step, plus `delete`/`push`/`pop`/`++`/`--`, are
/// writes through the pointed-to storage.
///
/// Parameters feed two consumers. Direct writes mark the parameter itself as
/// structural, while one level of forwarding to a directly-mutating callee
/// propagates that requirement through wrappers such as EnumerableSet's
/// `Bytes32Set._inner` adapters. The direct-only callee scan deliberately does
/// not recurse: passing a pointer to a view/pure helper is not evidence of a
/// write, and recursive wrapper cycles must not manufacture one.
struct StorageRefWriteThroughScanner: ASTConstVisitor
{
	std::set<VariableDeclaration const*> writtenThrough;
	bool followCalls = true;

	explicit StorageRefWriteThroughScanner(bool _followCalls = true): followCalls(_followCalls) {}

	/// Record `_lhs`'s peeled root when it is a storage-located local/parameter
	/// declaration. `_requireSteps` distinguishes an assignment (which must
	/// have at least one access step to be a write THROUGH the pointer) from
	/// `delete`/`push`/`pop`, which write through even on a bare identifier.
	void noteRoot(Expression const& _lhs, bool _requireSteps)
	{
		Expression const* root = peelLValueRoot(_lhs);
		if (_requireSteps && root == &_lhs)
			return;
		auto const* ident = dynamic_cast<Identifier const*>(root);
		if (!ident)
			return;
		auto const* decl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration);
		if (!decl || decl->isStateVariable())
			return;
		if (decl->referenceLocation() == VariableDeclaration::Location::Storage)
			writtenThrough.insert(decl);
	}

	bool visit(Assignment const& _assignment) override
	{
		// Tuple-destructuring LHS components are individual lvalues; a
		// component writing through a storage pointer is exactly as dropped
		// as a standalone assignment (mirrors StorageRefRebindScanner).
		if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_assignment.leftHandSide()))
		{
			for (auto const& component: tuple->components())
				if (component)
					noteRoot(*component, /*_requireSteps=*/true);
		}
		else
			noteRoot(_assignment.leftHandSide(), /*_requireSteps=*/true);
		return true;
	}

	bool visit(UnaryOperation const& _unary) override
	{
		// `delete p` on a storage pointer clears the POINTED-TO storage, so
		// it is a write through even with no access steps. `p[i]++` / `p.f--`
		// are ordinary read-modify-writes of the pointed-to location.
		if (_unary.getOperator() == Token::Delete)
			noteRoot(_unary.subExpression(), /*_requireSteps=*/false);
		else if (_unary.getOperator() == Token::Inc || _unary.getOperator() == Token::Dec)
			noteRoot(_unary.subExpression(), /*_requireSteps=*/true);
		return true;
	}

	bool visit(FunctionCall const& _call) override
	{
		auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_call.expression());
		if (memberAccess && (memberAccess->memberName() == "push" || memberAccess->memberName() == "pop"))
			noteRoot(memberAccess->expression(), /*_requireSteps=*/false);
		if (!followCalls)
			return true;


		Declaration const* referenced = nullptr;
		if (auto const* identifier = dynamic_cast<Identifier const*>(&_call.expression()))
			referenced = identifier->annotation().referencedDeclaration;
		else if (memberAccess)
			referenced = memberAccess->annotation().referencedDeclaration;
		auto const* callee = dynamic_cast<FunctionDefinition const*>(referenced);
		if (!callee || !callee->isImplemented())
			return true;
		// Propagate only compiler-proven direct writes from the callee. This
		// keeps ordinary read-only storage helpers on their value carrier.
		StorageRefWriteThroughScanner directWrites(/*_followCalls=*/false);
		callee->body().accept(directWrites);

		size_t parameterOffset = 0;
		auto const* functionType = dynamic_cast<FunctionType const*>(_call.expression().annotation().type);
		if (memberAccess && functionType && functionType->hasBoundFirstArgument())
		{
			VariableDeclaration const* parameter = callee->parameters().empty() ? nullptr : callee->parameters().front().get();
			if (parameter && directWrites.writtenThrough.count(parameter))
				noteRoot(memberAccess->expression(), /*_requireSteps=*/false);
			parameterOffset = 1;
		}
		for (size_t i = 0; i < _call.arguments().size() && i + parameterOffset < callee->parameters().size(); ++i)
		{
			VariableDeclaration const* parameter = callee->parameters()[i + parameterOffset].get();
			if (directWrites.writtenThrough.count(parameter))
				noteRoot(*_call.arguments()[i], /*_requireSteps=*/false);
		}
		return true;
	}
};

/// [P0 fail-closed: E0(a')] Called from the variable-declaration lowering at the
/// point where a storage-located declaration has NOT been registered as a
/// tracked alias and is about to fall through to the copy-`let` lowering.
/// Refuses loudly when the function later writes through that local, instead of
/// emitting a dead value-copy rebind and dropping the state mutation.
void requireNoWriteThroughUntrackedStorageLocalOrThrow(VariableDeclaration const& _decl)
{
	if (!storageRefWriteThroughLocals.count(&_decl))
		return;
	throw UnsupportedSolCore(
		"Write through the untracked storage-pointer local `" + _decl.name()
		+ "` is not "
		  "modeled: its initializer does not resolve to a storage path (a call-returned "
		  "reference, or a pointer this function rebinds), so the declaration lowers to a "
		  "copy of the pointed-to VALUE and a later `"
		+ _decl.name()
		+ ".…= v` / `delete` / "
		  "`push`/`pop` would be exported as a rebind of that dead copy — leaving storage "
		  "unchanged while the real EVM writes through the pointer. Refusing instead of "
		  "silently dropping the state mutation.");
}

/// [P0 fail-closed: storage-ref early binding vs. array shrink]
///
/// Whole-function pre-pass for storage-ref early binding versus carrier-
/// replacing shrink operations.
///
/// Dynamic storage-array pop and direct delete are deliberately excluded once
/// they use the slot-preserving StorageArray path: pop clears the old live slot,
/// delete clears the old live prefix, and both retain the SparseMap carrier, so
/// a rooted stale reference can continue to read/write its captured slot.
/// Aggregate delete, whole reassignment, unresolved or recursive array shapes,
/// and element types containing mappings remain here because they still lack
/// that complete structural preservation path.
///
/// Roots are matched by state-variable name. An uncovered shrink whose root
/// cannot be resolved to a state variable sets `anyUnresolvedShrink`, which
/// poisons every array-indexed bind in the function.
bool arrayElementContainsRecursiveStruct(ArrayType const& _arrayType)
{
	Type const* elementType = _arrayType.baseType();
	while (auto const* nestedArray = dynamic_cast<ArrayType const*>(elementType))
		elementType = nestedArray->baseType();
	auto const* structType = dynamic_cast<StructType const*>(elementType);
	return structType && structType->recursive();
}

bool hasSlotPreservingDynamicStorageArraySemantics(Type const* _type)
{
	auto const* arrayType = dynamic_cast<ArrayType const*>(_type);
	return arrayType && arrayType->location() == DataLocation::Storage && arrayType->isDynamicallySized()
		   && !arrayType->containsNestedMapping() && !arrayElementContainsRecursiveStruct(*arrayType);
}

struct StorageRefShrinkScanner: ASTConstVisitor
{
	StorageRefShrinkInfo info;

	static bool isStorageLocated(Type const* _type) { return _type && _type->dataStoredIn(DataLocation::Storage); }

	static std::optional<std::string> stateVarRootName(Expression const& _expr)
	{
		if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
		{
			auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
			if (decl && decl->isStateVariable())
				return decl->name();
			return std::nullopt;
		}
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
			return stateVarRootName(memberAccess->expression());
		if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr))
			return stateVarRootName(indexAccess->baseExpression());
		return std::nullopt;
	}

	void noteShrink(Expression const& _base)
	{
		if (auto rootName = stateVarRootName(_base))
			info.shrunkRoots.insert(*rootName);
		else
			info.anyUnresolvedShrink = true;
	}

	bool visit(FunctionCall const& _call) override
	{
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_call.expression()))
			if (memberAccess->memberName() == "pop"
				&& !hasSlotPreservingDynamicStorageArraySemantics(memberAccess->expression().annotation().type))
				noteShrink(memberAccess->expression());
		return true;
	}

	bool visit(UnaryOperation const& _unary) override
	{
		Type const* targetType = _unary.subExpression().annotation().type;
		if (_unary.getOperator() == Token::Delete && isStorageLocated(targetType)
			&& !hasSlotPreservingDynamicStorageArraySemantics(targetType))
			noteShrink(_unary.subExpression());
		return true;
	}

	bool visit(Assignment const& _assignment) override
	{
		if (isStorageLocated(_assignment.leftHandSide().annotation().type))
			noteShrink(_assignment.leftHandSide());
		return true;
	}
};

/// Narrow-`bytesN` width convention (SolCoreBase.ml [BytesW] doc): the
/// SolCore value of a `bytesN` is carried RIGHT-aligned (the numeric value of
/// its N bytes). Under that carrier an implicit Solidity `bytesN -> bytesM`
/// WIDENING (M > N, including M = 32) — a register no-op on the EVM, hence
/// syntactically invisible in the AST — is a real `<< 8*(M-N)` the export
/// must materialize. This scanner enumerates every implicit-conversion site
/// solc's TypeChecker can introduce (assignment RHS, declaration
/// initializers, returns, call/emit/revert/struct-ctor arguments,
/// inline-array components, mapping keys) and FAILS CLOSED on any
/// widening the expression lowering does not explicitly materialize, instead
/// of silently exporting a value that every observation would mis-align.
/// This walk also records the structural target type of every byte/string
/// literal. The literal's own annotation deliberately does not choose among
/// `string`, `bytes`, and `bytesN`; the surrounding typed flow does. The map
/// is consumed directly by `exportExpr`, never reconstructed from names or
/// from the literal payload.
struct NarrowBytesWideningScanner: ASTConstVisitor
{
	static void checkFlow(Expression const& _source, Type const* _target)
	{
		if (!_target)
			return;
		if (auto const* literal = dynamic_cast<Literal const*>(&_source))
			if (dynamic_cast<StringLiteralType const*>(literal->annotation().type))
			{
				canonicalLiteralTargets[literal] = _target;
				return;
			}
		if (auto const* conditional = dynamic_cast<Conditional const*>(&_source))
			// A fixed-bytes conditional materializes each branch at its own
			// compiler-resolved common type in exportExpr. Keep checking the
			// conditional itself below so an additional outer widening still
			// fails closed at this non-materializing flow site.
			if (!dynamic_cast<FixedBytesType const*>(conditional->annotation().type))
			{
				checkFlow(conditional->trueExpression(), _target);
				checkFlow(conditional->falseExpression(), _target);
			}

		auto const* targetFB = dynamic_cast<FixedBytesType const*>(_target);
		if (!targetFB)
			return;
		Type const* sourceType = _source.annotation().type;
		if (!sourceType)
			return;
		if (auto const* sourceFB = dynamic_cast<FixedBytesType const*>(sourceType))
			if (sourceFB->numBytes() < targetFB->numBytes())
				throw UnsupportedSolCore(
					"Implicit bytes" + std::to_string(sourceFB->numBytes()) + " -> bytes"
					+ std::to_string(targetFB->numBytes())
					+ " widening at a site whose lowering does not materialize the "
					  "left-shift the narrow-bytesN right-aligned value convention "
					  "requires. Refusing rather than exporting a value every "
					  "observation would mis-align; write the widening as an "
					  "explicit cast (lowered as a shift) instead.");
	}

	bool visit(Assignment const& _assignment) override
	{
		// Compound assignments (`|=`, `^=`, `<<=`, ...) type-check the
		// operation at the LHS type, so a genuinely narrower RHS is an
		// operand widening — same rule as plain `=`.
		if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_assignment.leftHandSide()))
		{
			auto const* rhsTuple = dynamic_cast<TupleExpression const*>(&_assignment.rightHandSide());
			if (rhsTuple && !rhsTuple->isInlineArray() && rhsTuple->components().size() == tuple->components().size())
			{
				for (size_t i = 0; i < tuple->components().size(); ++i)
					if (tuple->components()[i] && rhsTuple->components()[i])
						checkFlow(*rhsTuple->components()[i], tuple->components()[i]->annotation().type);
			}
			// Non-tuple RHS (function call returning a tuple): component
			// types come from the callee's declared returns, which the
			// FunctionCall visit below cannot see per-component; the
			// callee's own Return sites are checked in its own export, and
			// the components here are already converted values.
		}
		else
			checkFlow(_assignment.rightHandSide(), _assignment.leftHandSide().annotation().type);
		return true;
	}

	bool visit(VariableDeclarationStatement const& _stmt) override
	{
		Expression const* init = _stmt.initialValue();
		if (!init)
			return true;
		auto const& decls = _stmt.declarations();
		auto const* initTuple = dynamic_cast<TupleExpression const*>(init);
		if (initTuple && !initTuple->isInlineArray() && initTuple->components().size() == decls.size()
			&& decls.size() > 1)
		{
			for (size_t i = 0; i < decls.size(); ++i)
				if (decls[i] && initTuple->components()[i])
					checkFlow(*initTuple->components()[i], decls[i]->type());
		}
		else if (decls.size() == 1 && decls.front())
			checkFlow(*init, decls.front()->type());
		return true;
	}

	bool visit(Return const& _return) override
	{
		// The type checker records the governing return parameter list on
		// the node itself, which is also correct for returns inside
		// modifier bodies.
		ParameterList const* paramList = _return.annotation().functionReturnParameters;
		if (!_return.expression() || !paramList)
			return true;
		auto const& retParams = paramList->parameters();
		auto const* tuple = dynamic_cast<TupleExpression const*>(_return.expression());
		if (tuple && !tuple->isInlineArray() && tuple->components().size() == retParams.size() && retParams.size() > 1)
		{
			for (size_t i = 0; i < retParams.size(); ++i)
				if (tuple->components()[i])
					checkFlow(*tuple->components()[i], retParams[i]->type());
		}
		else if (retParams.size() == 1)
			checkFlow(*_return.expression(), retParams.front()->type());
		return true;
	}

	bool visit(FunctionCall const& _call) override
	{
		if (*_call.annotation().kind == FunctionCallKind::TypeConversion)
		{
			// Explicit casts are materialized by exportExpr. Record their exact
			// target before descending so byte/string literals retain whether
			// the cast chose `string`, dynamic `bytes`, or fixed `bytesN`.
			if (!_call.arguments().empty())
				if (auto const* literal = dynamic_cast<Literal const*>(_call.arguments().front().get()))
					if (dynamic_cast<StringLiteralType const*>(literal->annotation().type))
						canonicalLiteralTargets[literal] = _call.annotation().type;
			return true;
		}
		auto const* funType = dynamic_cast<FunctionType const*>(_call.expression().annotation().type);
		if (!funType)
			return true;
		if (funType->kind() == FunctionType::Kind::BytesConcat || funType->kind() == FunctionType::Kind::StringConcat)
		{
			Type const* target = funType->kind() == FunctionType::Kind::BytesConcat
									 ? static_cast<Type const*>(TypeProvider::bytesMemory())
									 : static_cast<Type const*>(TypeProvider::stringMemory());
			for (auto const& arg: _call.sortedArguments())
				if (arg)
					checkFlow(*arg, target);
			return true;
		}
		switch (funType->kind())
		{
		case FunctionType::Kind::ABIEncode:
		case FunctionType::Kind::ABIEncodePacked:
		case FunctionType::Kind::ABIEncodeWithSelector:
		case FunctionType::Kind::ABIEncodeWithSignature:
		case FunctionType::Kind::ABIEncodeCall:
		case FunctionType::Kind::ABIDecode:
			// Variadic ABI builtins encode each argument at its own literal
			// type, so no contextual target is introduced here.
			return true;
		default:
			break;
		}
		auto const& paramTypes = funType->parameterTypes();
		auto const& args = _call.sortedArguments();
		if (paramTypes.size() != args.size())
			return true; // e.g. bound calls / value options; conservative skip
		for (size_t i = 0; i < args.size(); ++i)
			if (args[i])
				checkFlow(*args[i], paramTypes[i]);
		return true;
	}

	bool visit(Conditional const& _conditional) override
	{
		// exportExprCoercedToFixedBytes owns branch widening when solc has
		// resolved a fixed-bytes common type for the conditional.
		if (!dynamic_cast<FixedBytesType const*>(_conditional.annotation().type))
		{
			checkFlow(_conditional.trueExpression(), _conditional.annotation().type);
			checkFlow(_conditional.falseExpression(), _conditional.annotation().type);
		}
		return true;
	}

	bool visit(TupleExpression const& _tuple) override
	{
		if (!_tuple.isInlineArray())
			return true;
		auto const* arrayType = dynamic_cast<ArrayType const*>(_tuple.annotation().type);
		if (!arrayType)
			return true;
		for (auto const& component: _tuple.components())
			if (component)
				checkFlow(*component, arrayType->baseType());
		return true;
	}

	bool visit(IndexAccess const& _access) override
	{
		if (!_access.indexExpression())
			return true;
		if (auto const* mappingType = dynamic_cast<MappingType const*>(_access.baseExpression().annotation().type))
			checkFlow(*_access.indexExpression(), mappingType->keyType());
		return true;
	}
};

/// True iff binding a storage-ref alias to `_target` would put the model's
/// live re-projection at odds with the EVM's early slot binding, because the
/// path steps through an array index into a root that the function can shrink.
bool storageRefTargetRacesArrayShrink(StorageRefTarget const& _target)
{
	bool indexesAnArray = false;
	for (StorageRefStep const& step: _target.steps)
		if (step.kind == StorageRefStep::Kind::ArrayIndex)
			indexesAnArray = true;
	if (!indexesAnArray)
		return false;
	return storageRefShrinkInfo.anyUnresolvedShrink || storageRefShrinkInfo.shrunkRoots.count(_target.root) > 0;
}

// --- End general storage-reference-variable alias tracking (data model) ---

Json exporterMetadata(std::string const& _contractName, bool _viaIR)
{
	Json metadata = Json::object();
	metadata["schemaVersion"] = "0.2.0";
	metadata["compilerVersion"] = VersionString;
	metadata["contract"] = _contractName;
	// Which codegen pipeline compiled this unit. solc's legacy and via-IR
	// pipelines DIVERGE on intra-statement evaluation order (measured on a
	// real EVM: binary operands legacy RIGHT-first / via-IR LEFT-first;
	// indexed event args legacy REVERSE-source-order / via-IR source order),
	// so any consumer that commits to one pipeline's order (the Class C-pin
	// eval-order table in the aeneas generator) must fail closed unless this
	// field names the pipeline its table row was verified against. Absent
	// field (older artifact) must be treated as UNKNOWN, never defaulted.
	metadata["codegen"] = _viaIR ? "via-ir" : "legacy";
	if (VersionCompactBytes.size() >= 2)
		metadata["exporterFamily"] = "solcore-solidity-"s
									 + std::to_string(static_cast<unsigned>(VersionCompactBytes[0])) + "."
									 + std::to_string(static_cast<unsigned>(VersionCompactBytes[1]));
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
	// Width-dependent unsigned arithmetic/shift nodes carry an explicit
	// 8..256 `bits` value. The historical flag remains for producer-family
	// continuity; explicitArithmeticMetadata pins the strict cutover.
	flags["arithWidths"] = true;
	flags["explicitArithmeticMetadata"] = true;
	// String/hex/unicode literals use the canonical typed_literal envelope.
	// Consumers reject artifacts without this producer-owned cutover marker
	// instead of accepting the old scalarized `string`/`u256` payloads.
	flags["canonicalLiterals"] = true;
	// General local storage-reference-variable alias tracking (design
	// §3.5): artifacts with this flag substitute live rooted storage
	// reads/writes for resolved `T storage x = <storage lvalue>;` aliases
	// (and lower alias push/pop to array_push_expr/array_pop_expr);
	// artifacts without it predate the mechanism and model such aliases
	// as bind-time value copies rescued only by the write oracle's
	// fail-closed `unknown`.
	flags["storageRefAliases"] = true;
	// Signed integer comparison/shift/div/mod/add/sub/mul/neg/signextend
	// vocabulary (i256_lt/le/gt/ge/sar/div/mod/add/sub/mul/neg,
	// signextend). Distinct `i256_*`/`signextend` kinds, never a flag on
	// the old `u256_*` kinds -- an old (pre-this-flag) frontend fails
	// closed on the new kinds via its existing unknown-kind `Parse_error`
	// path, so this flag is purely informational for consumers that want
	// to detect artifacts carrying the new vocabulary.
	flags["signedOps"] = true;
	// msg.sig support: the exported CallEnv decl carries an 8th field,
	// `msgSig` (u256, right-aligned 4-byte selector of the current call
	// frame). Artifacts without this flag predate the field and have the
	// pre-existing 6/7-field CallEnv shape.
	flags["msgSig"] = true;
	// `**`/`<<` vocabulary. Width is explicit even at 256 and arithmetic
	// checkedness is always a bool; missing/null metadata is rejected.
	flags["expShlWidths"] = true;
	// Statically-bound super/Base calls carry an exact flattened coordinate.
	flags["staticBaseCallTargets"] = true;
	// The exact virtual-slot inventory is present even when empty.
	flags["unboundVirtualSlots"] = true;
	// Every `.balance` access lowers to a canonical `balance_of` world read,
	// including the current contract's address.
	flags["balanceOf"] = true;
	// EIP-1153 `transient` state variables: full-slot value-typed transient
	// vars export as the crate-level `transient_state` list with dedicated
	// transient_get/transient_set nodes (never Storage fields); every other
	// transient shape still fails the contract's export loudly.
	flags["transientState"] = true;
	// Structural metatype code members are exported without byte payloads.
	// Bound consumers obtain the exact bytes from the sibling artifact
	// manifest; unbound consumers keep them opaque.
	flags["contractCodeIntrospection"] = true;
	return flags;
}

Json unsupportedExport(std::string const& _contractName, std::string const& _reason, bool _viaIR)
{
	Json result = Json::object();
	result["unsupported"] = true;
	result["reason"] = _reason;
	Json metadata = exporterMetadata(_contractName, _viaIR);
	for (auto const& [key, value]: metadata.items())
		result[key] = value;
	if (!_viaIR)
		result["evalOrderCommitment"] = EvalOrderCommitment;
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

std::string exportedContractId(ContractDefinition const& _contract) { return _contract.fullyQualifiedName(); }

Json canonicalStringTypedLiteral(std::string const& _value)
{
	static char const digits[] = "0123456789abcdef";
	std::string sourceBytes;
	sourceBytes.reserve(_value.size() * 2);
	for (char byte: _value)
	{
		sourceBytes += digits[static_cast<unsigned char>(byte) >> 4];
		sourceBytes += digits[static_cast<unsigned char>(byte) & 0xf];
	}
	Json target = Json::object();
	target["kind"] = "string";
	Json result = Json::object();
	result["kind"] = "typed_literal";
	result["encoding"] = "plain";
	result["sourceBytes"] = std::move(sourceBytes);
	result["width"] = _value.size();
	result["target"] = std::move(target);
	return result;
}

// The runtime implements only this fixed set of read-only Checkpoints queries.
// Do not infer this tag from the exported function name: a library with a
// matching name/signature is not an OpenZeppelin Checkpoints declaration unless
// its resolved AST declaration also belongs to one of the canonical OZ sources.
bool isKnownOzCheckpointsQuery(FunctionDefinition const& _function)
{
	auto const* library = dynamic_cast<ContractDefinition const*>(_function.scope());
	if (!library || !library->isLibrary() || _function.annotation().contract != library || !_function.isOrdinary()
		|| !_function.isImplemented() || _function.visibility() != Visibility::Internal
		|| _function.stateMutability() != StateMutability::View)
		return false;

	std::string const sourceName = library->sourceUnitName();
	bool const legacySource = sourceName == "@openzeppelin/contracts/utils/Checkpoints.sol"
							  || sourceName == "@openzeppelin/contracts-upgradeable/utils/CheckpointsUpgradeable.sol";
	bool const modernSource = sourceName == "@openzeppelin/contracts/utils/structs/Checkpoints.sol";
	bool const upgradeableSource = sourceName == "@openzeppelin/contracts-upgradeable/utils/CheckpointsUpgradeable.sol";
	std::string const expectedLibraryName = upgradeableSource ? "CheckpointsUpgradeable" : "Checkpoints";
	if ((!legacySource && !modernSource) || library->name() != expectedLibraryName
		|| library->fullyQualifiedName() != sourceName + ":" + expectedLibraryName)
		return false;

	auto const& sourceUnit = library->sourceUnit();
	auto hasExpectedSourceLocation = [&sourceName](ASTNode const& _node)
	{ return _node.location().sourceName && *_node.location().sourceName == sourceName; };
	if (&sourceUnit != &_function.sourceUnit() || !sourceUnit.location().sourceName
		|| *sourceUnit.location().sourceName != sourceName || !hasExpectedSourceLocation(sourceUnit)
		|| !hasExpectedSourceLocation(*library) || !hasExpectedSourceLocation(_function)
		|| !sourceUnit.location().contains(library->location())
		|| !sourceUnit.location().contains(_function.location()))
		return false;

	auto const& parameters = _function.parameters();
	auto const& returns = _function.returnParameters();
	if (parameters.empty() || parameters.front()->referenceLocation() != VariableDeclaration::Location::Storage
		|| returns.size() != 1)
		return false;

	auto const* selfType = dynamic_cast<StructType const*>(parameters.front()->type());
	if (!selfType)
		return false;
	StructDefinition const& selfStruct = selfType->structDefinition();
	if (selfStruct.scope() != library || &selfStruct.sourceUnit() != &sourceUnit
		|| selfStruct.sourceUnitName() != sourceName || !hasExpectedSourceLocation(selfStruct)
		|| !sourceUnit.location().contains(selfStruct.location()))
		return false;

	unsigned keyBits = 0;
	unsigned valueBits = 0;
	bool const isHistory = selfStruct.name() == "History" && legacySource;
	if (isHistory)
	{
		keyBits = 32;
		valueBits = 224;
	}
	else if (selfStruct.name() == "Trace224")
	{
		keyBits = 32;
		valueBits = 224;
	}
	else if (selfStruct.name() == "Trace208" && modernSource)
	{
		keyBits = 48;
		valueBits = 208;
	}
	else if (selfStruct.name() == "Trace256" && modernSource)
	{
		keyBits = 256;
		valueBits = 256;
	}
	else if (selfStruct.name() == "Trace160")
	{
		keyBits = 96;
		valueBits = 160;
	}
	else
		return false;

	auto isUnsignedInteger = [](Type const* _type, unsigned _bits)
	{
		auto const* integer = dynamic_cast<IntegerType const*>(_type);
		return integer && !integer->isSigned() && integer->numBits() == _bits;
	};

	std::string const& functionName = _function.name();
	if (functionName == "latest")
		return parameters.size() == 1
			   && (isHistory ? (isUnsignedInteger(returns.front()->type(), 224)
								|| isUnsignedInteger(returns.front()->type(), 256))
							 : isUnsignedInteger(returns.front()->type(), valueBits));
	if (functionName == "length")
		return parameters.size() == 1 && isUnsignedInteger(returns.front()->type(), 256);
	if (functionName != "lowerLookup" && functionName != "upperLookup" && functionName != "upperLookupRecent")
		return false;
	return parameters.size() == 2 && isUnsignedInteger(parameters[1]->type(), keyBits)
		   && isUnsignedInteger(returns.front()->type(), valueBits);
}

void addInternalLibraryCallContractId(Json& _result, FunctionDefinition const& _function)
{
	auto const* library = dynamic_cast<ContractDefinition const*>(_function.scope());
	if (!library || !library->isLibrary())
		return;

	_result["contractId"] = exportedContractId(*library);
	if (isKnownOzCheckpointsQuery(_function))
		_result["runtimeKind"] = "oz_checkpoints_query";
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
	if (label == "uint256" || label == "int256"
		|| (label.size() > 4 && (label.substr(0, 4) == "uint" || label.substr(0, 3) == "int"))
		|| (label.size() > 5 && label.substr(0, 5) == "bytes"))
		return Json("u256");
	return std::nullopt;
}

bool exportStorageLayoutType(Json const& _types, std::string const& _typeId, Json& _valueType, Json& _keys)
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
	CompilerStack const& _compilerStack, ContractDefinition const& _contract, std::string const& _fieldName)
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

/// A `transient`-location state variable (EIP-1153, solc 0.8.28+). Used to
/// route reads/writes to the dedicated transient_get/transient_set nodes
/// instead of the persistent Storage model. The AST declaration itself
/// carries the location, so call sites need no side registry.
bool isTransientStateVar(VariableDeclaration const* _decl)
{
	return _decl && _decl->isStateVariable() && _decl->referenceLocation() == VariableDeclaration::Location::Transient;
}

/// Slot + intra-slot byte offset of a transient state variable, from solc's
/// own transient storage layout (same JSON shape as the persistent
/// storageLayout, separate EIP-1153 address space). nullopt when the layout
/// is unavailable or the variable is not in it — callers must fail closed,
/// never guess a slot.
std::optional<std::pair<std::string, int>> lookupTransientSlotOffset(
	CompilerStack const& _compilerStack, ContractDefinition const& _contract, std::string const& _fieldName)
{
	try
	{
		Json const& layout = _compilerStack.transientStorageLayout(_contract.fullyQualifiedName());
		for (auto const& entry: layout.at("storage"))
			if (entry.value("label", "") == _fieldName)
				return std::make_pair(entry.value("slot", std::string("0")), entry.value("offset", 0));
	}
	catch (...)
	{
	}
	return std::nullopt;
}

std::optional<int> functionParameterIndex(FunctionDefinition const& _function, Declaration const* _declaration)
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
	std::vector<int>& _keyArgOrder)
{
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
	{
		auto const* variable = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
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
		if (!collectStorageGetterAccess(indexAccess->baseExpression(), _function, _stateVar, _keyArgOrder))
			return false;
		auto const* keyIdentifier = dynamic_cast<Identifier const*>(indexAccess->indexExpression());
		if (!keyIdentifier)
			return false;
		auto paramIndex = functionParameterIndex(_function, keyIdentifier->annotation().referencedDeclaration);
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
	std::vector<int> const& _keyArgOrder)
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
	CompilerStack const& _compilerStack, ContractDefinition const& _contract, FunctionDefinition const& _function)
{
	if (!_function.isImplemented())
		return std::nullopt;
	if (_function.body().statements().size() != 1)
		return std::nullopt;
	auto const* returnStmt = dynamic_cast<Return const*>(_function.body().statements().front().get());
	if (!returnStmt || !returnStmt->expression())
		return std::nullopt;
	VariableDeclaration const* stateVar = nullptr;
	std::vector<int> keyArgOrder;
	if (!collectStorageGetterAccess(*returnStmt->expression(), _function, stateVar, keyArgOrder))
		return std::nullopt;
	if (!stateVar)
		return std::nullopt;
	return exportStorageGetterResolution(_compilerStack, _contract, *stateVar, keyArgOrder);
}

Json exportResolvedType(Type const* _type, bool _storage = false);

Json exportByteHelperArgumentType(Type const* _type)
{
	if (
		_type
		&& (
			_type->category() == Type::Category::StringLiteral
			|| (
				_type->category() == Type::Category::Array
				&& dynamic_cast<ArrayType const*>(_type)->isByteArrayOrString()
			)
		)
	)
	{
		Json byteArrayType = Json::object();
		byteArrayType["kind"] = "named";
		byteArrayType["name"] = "ByteArray";
		return byteArrayType;
	}
	return exportResolvedType(_type);
}

Json dataLocationEntry(VariableDeclaration const& _decl);

std::optional<Json> exportReturnType(TypePointers const& _returns)
{
	if (_returns.empty())
		return std::nullopt;
	if (_returns.size() == 1)
		return exportResolvedType(_returns.front());

	Json elements = Json::array();
	for (Type const* returnType: _returns)
		elements.emplace_back(exportResolvedType(returnType));
	Json tupleType = Json::object();
	tupleType["kind"] = "tuple";
	tupleType["elements"] = std::move(elements);
	return tupleType;
}

Json exportAbiDescriptor(
	std::string const& _name,
	Type const* _encodingType,
	Type const* _solidityType,
	bool _forLibrary)
{
	if (!_encodingType || !_solidityType)
		throw UnsupportedSolCore("ABI descriptor requires compiler-resolved encoding and Solidity types.");

	Json result = Json::object();
	result["name"] = _name;
	result["internalType"] = _solidityType->toString(true);
	result["components"] = Json::array();

	bool const libraryStoragePointer
		= _forLibrary && _encodingType->dataStoredIn(DataLocation::Storage);
	if (_encodingType->isValueType() || libraryStoragePointer)
	{
		// A library's interface type deliberately retains nominal enums for
		// selector construction. ABI descriptors describe the wire, however,
		// so value leaves must use the compiler's encoding type. Keep the
		// library-only storage-reference spelling: it transports a slot word
		// while retaining the referent needed by the consumer's storage model.
		Type const* wireType = libraryStoragePointer ? _encodingType : _encodingType->encodingType();
		if (!wireType)
			throw UnsupportedSolCore("Compiler-resolved ABI value type has no encoding type.");
		std::string type = wireType->canonicalName();
		if (type.empty())
			throw UnsupportedSolCore("Compiler-resolved ABI type has no canonical spelling.");
		result["type"] = type + (libraryStoragePointer ? " storage" : "");
		return result;
	}

	if (auto const* arrayType = dynamic_cast<ArrayType const*>(_encodingType))
	{
		if (arrayType->isByteArrayOrString())
		{
			result["type"] = arrayType->canonicalName();
			return result;
		}

		auto const* solidityArrayType = dynamic_cast<ArrayType const*>(_solidityType);
		if (!solidityArrayType || !arrayType->baseType() || !solidityArrayType->baseType())
			throw UnsupportedSolCore("Compiler-resolved ABI array lost its Solidity element type.");
		// Inside a wire-encoded array the elements are encoded inline; the
		// library-only storage-pointer spelling applies to top-level
		// parameters only (storage aggregates never reach this branch).
		Json element = exportAbiDescriptor(
			"",
			arrayType->baseType(),
			solidityArrayType->baseType(),
			false);
		if (!element["type"].is_string() || !element["components"].is_array())
			throw UnsupportedSolCore("Compiler-resolved ABI array element has a malformed descriptor.");
		std::string suffix = arrayType->isDynamicallySized()
								 ? "[]"
								 : "[" + arrayType->length().str() + "]";
		result["type"] = element["type"].get<std::string>() + suffix;
		result["components"] = std::move(element["components"]);
		return result;
	}

	if (auto const* structType = dynamic_cast<StructType const*>(_encodingType))
	{
		auto const* solidityStructType = dynamic_cast<StructType const*>(_solidityType);
		if (
			!solidityStructType
			|| &structType->structDefinition() != &solidityStructType->structDefinition()
		)
			throw UnsupportedSolCore("Compiler-resolved ABI tuple lost its Solidity struct identity.");
		result["type"] = "tuple";
		for (auto const& member: solidityStructType->structDefinition().members())
		{
			Type const* solidityMemberType = member->annotation().type;
			if (!solidityMemberType)
				throw UnsupportedSolCore("ABI struct member has no compiler-resolved Solidity type.");
			// Struct member types default to storage locations in the AST,
			// and a library's interfaceType(true) preserves them, which
			// would misrender wire-tuple members as componentless storage
			// pointers. This tuple is wire-encoded (storage-pointer params
			// return before recursion), so members encode as ordinary ABI.
			Type const* encodingMemberType = solidityMemberType->interfaceType(false);
			if (!encodingMemberType)
				throw UnsupportedSolCore(
					"ABI struct member '" + member->name() + "' has no compiler-resolved external type.");
			result["components"].emplace_back(exportAbiDescriptor(
				member->name(),
				encodingMemberType,
				solidityMemberType,
				false));
		}
		return result;
	}

	if (auto const* tupleType = dynamic_cast<TupleType const*>(_encodingType))
	{
		auto const* solidityTupleType = dynamic_cast<TupleType const*>(_solidityType);
		if (!solidityTupleType || tupleType->components().size() != solidityTupleType->components().size())
			throw UnsupportedSolCore("Compiler-resolved ABI tuple has mismatched component types.");
		result["type"] = "tuple";
		for (size_t i = 0; i < tupleType->components().size(); ++i)
		{
			if (!tupleType->components()[i] || !solidityTupleType->components()[i])
				throw UnsupportedSolCore("Compiler-resolved ABI tuple contains an omitted component.");
			result["components"].emplace_back(exportAbiDescriptor(
				"",
				tupleType->components()[i],
				solidityTupleType->components()[i],
				false));
		}
		return result;
	}

	throw UnsupportedSolCore(
		"Compiler-resolved type '" + _solidityType->toString(true)
		+ "' has no supported canonical external ABI descriptor.");
}

Json exportAbiDescriptor(std::string const& _name, Type const* _solidityType, bool _forLibrary)
{
	if (!_solidityType)
		throw UnsupportedSolCore("ABI descriptor requires a compiler-resolved Solidity type.");
	if (auto const* tupleType = dynamic_cast<TupleType const*>(_solidityType))
	{
		Json result = Json::object();
		result["name"] = _name;
		result["type"] = "tuple";
		result["internalType"] = tupleType->toString(true);
		result["components"] = Json::array();
		for (Type const* component: tupleType->components())
		{
			if (!component)
				throw UnsupportedSolCore("Compiler-resolved ABI tuple contains an omitted component.");
			result["components"].emplace_back(exportAbiDescriptor("", component, _forLibrary));
		}
		return result;
	}
	Type const* encodingType = _solidityType->interfaceType(_forLibrary);
	if (!encodingType)
		throw UnsupportedSolCore(
			"Compiler-resolved type '" + _solidityType->toString(true)
			+ "' is not available at an external ABI boundary.");
	return exportAbiDescriptor(_name, encodingType, _solidityType, _forLibrary);
}

struct FunctionAbiDescriptors
{
	Json params;
	Json returns;
};

FunctionAbiDescriptors exportFunctionAbiDescriptors(FunctionType const& _funType)
{
	if (!_funType.hasDeclaration())
		throw UnsupportedSolCore("External ABI function metadata has no compiler-resolved declaration.");
	FunctionTypePointer interfaceType = _funType.interfaceFunctionType();
	if (!interfaceType)
		throw UnsupportedSolCore(
			"Function '" + _funType.declaration().name()
			+ "' has no compiler-resolved external ABI type.");

	bool forLibrary = false;
	if (auto const* contract = dynamic_cast<ContractDefinition const*>(_funType.declaration().scope()))
		forLibrary = contract->isLibrary();

	auto exportList = [&](std::vector<std::string> const& _names,
						  TypePointers const& _encodingTypes,
						  TypePointers const& _solidityTypes)
	{
		if (_names.size() != _encodingTypes.size() || _names.size() != _solidityTypes.size())
			throw UnsupportedSolCore(
				"Function '" + _funType.declaration().name()
				+ "' has inconsistent compiler-resolved ABI parameter metadata.");
		Json descriptors = Json::array();
		for (size_t i = 0; i < _names.size(); ++i)
			descriptors.emplace_back(exportAbiDescriptor(
				_names[i],
				_encodingTypes[i],
				_solidityTypes[i],
				forLibrary));
		return descriptors;
	};

	return FunctionAbiDescriptors{
		exportList(
			interfaceType->parameterNames(),
			interfaceType->parameterTypes(),
			_funType.parameterTypes()),
		exportList(
			interfaceType->returnParameterNames(),
			interfaceType->returnParameterTypes(),
			_funType.returnParameterTypes())};
}

Json exportForeignMethodSummary(FunctionTypePointer const& _funType)
{
	if (!_funType)
		throw UnsupportedSolCore("Foreign method summary has no compiler-resolved function type.");
	FunctionAbiDescriptors abi = exportFunctionAbiDescriptors(*_funType);
	Json method = Json::object();
	method["name"] = _funType->declaration().name();
	method["signature"] = _funType->externalSignature();
	method["selector"] = _funType->externalIdentifierHex();
	method["mutability"] = externalMutabilityString(_funType->stateMutability());
	method["paramsAbi"] = std::move(abi.params);
	method["returnsAbi"] = std::move(abi.returns);
	if (auto returnType = exportReturnType(_funType->returnParameterTypes()))
		method["return"] = std::move(*returnType);
	if (auto const* functionDef = dynamic_cast<FunctionDefinition const*>(&_funType->declaration()))
	{
		Json returnLocations = Json::array();
		bool anyLocation = false;
		for (auto const& retParam: functionDef->returnParameters())
		{
			Json location = retParam ? dataLocationEntry(*retParam) : Json();
			if (!location.is_null())
				anyLocation = true;
			returnLocations.emplace_back(std::move(location));
		}
		if (anyLocation)
			method["returnLocations"] = std::move(returnLocations);
	}
	return method;
}

Json exportRevertPayload(
	FunctionCall const& _call,
	bool _unconditionalPayload,
	bool _includeRuntimeArgs = true);

Json exportDispatchEntry(FunctionTypePointer const& _funType, FunctionDefinition const* _functionDef = nullptr)
{
	if (!_funType)
		throw UnsupportedSolCore("Dispatch entry has no compiler-resolved function type.");
	FunctionAbiDescriptors abi = exportFunctionAbiDescriptors(*_funType);
	Json entry = Json::object();
	entry["abiFunction"] = _funType->declaration().name();
	entry["function"] = _functionDef ? exportedFunctionName(*_functionDef) : _funType->declaration().name();
	entry["signature"] = _funType->externalSignature();
	entry["selector"] = _funType->externalIdentifierHex();
	entry["paramsAbi"] = std::move(abi.params);
	entry["returnsAbi"] = std::move(abi.returns);
	if (auto returnType = exportReturnType(_funType->returnParameterTypes()))
		entry["return"] = std::move(*returnType);
	else
		entry["return"] = "unit";
	if (_functionDef)
	{
		Json returnLocations = Json::array();
		bool anyLocation = false;
		for (auto const& retParam: _functionDef->returnParameters())
		{
			Json location = retParam ? dataLocationEntry(*retParam) : Json();
			if (!location.is_null())
				anyLocation = true;
			returnLocations.emplace_back(std::move(location));
		}
		if (anyLocation)
			entry["returnLocations"] = std::move(returnLocations);
	}

	// AUTO-GETTER SHALLOW OMISSION. For a public STATE VARIABLE the
	// FunctionType is FunctionType::FunctionType(VariableDeclaration const&)
	// (Types.cpp), whose return-parameter list is not the variable's type: for
	// a struct it is the struct's top-level members with mapping members
	// dropped and array members dropped UNLESS they are bytes/string, while
	// nested structs are returned WHOLE (a nested struct that itself contains
	// a mapping makes solc reject the public variable outright, so recursion
	// never arises). Carry solc's own kept-member NAMES, in solc's own order.
	//
	// The consumer cannot re-derive this rule: SolCore erases `bytes`/`string`
	// and `uint8[]` to the same array-of-u8 type, and solc KEEPS the former
	// while DROPPING the latter, so a consumer-side filter would silently
	// return a member solc omits — a wrong ABI no stage gate could see.
	// Emitted ONLY for the struct case; absent everywhere else, so every
	// existing consumer and artifact is unaffected.
	if (auto const* varDecl = dynamic_cast<VariableDeclaration const*>(&_funType->declaration()))
	{
		Type const* returnType = varDecl->annotation().type;
		// Mirror the index walk solc performs before looking at the struct:
		// mapping keys and (non-byte) array indices become getter PARAMETERS.
		while (returnType)
		{
			if (auto const* mappingType = dynamic_cast<MappingType const*>(returnType))
				returnType = mappingType->valueType();
			else if (auto const* arrayType = dynamic_cast<ArrayType const*>(returnType))
			{
				if (arrayType->isByteArrayOrString())
					break;
				returnType = arrayType->baseType();
			}
			else
				break;
		}
		if (dynamic_cast<StructType const*>(returnType))
		{
			Json members = Json::array();
			bool allNamed = true;
			for (std::string const& name: _funType->returnParameterNames())
			{
				if (name.empty())
				{
					allNamed = false;
					break;
				}
				members.emplace_back(name);
			}
			// An unnamed return parameter would make the member unaddressable
			// on the consumer side; omit the field entirely so the consumer
			// keeps its existing (loud) refusal rather than projecting a
			// guessed tuple. Struct members always have names, so this is a
			// defensive branch, not an expected one.
			if (allNamed)
				entry["getterReturnMembers"] = std::move(members);
		}
	}

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
						pushPayload(exportRevertPayload(
							_call,
							/*_unconditionalPayload=*/false,
							/*_includeRuntimeArgs=*/false));
					else if (callee->name() == "assert")
						pushPayload(Json::object());
				}
				if (dynamic_cast<ErrorDefinition const*>(calleeDecl))
					pushPayload(exportRevertPayload(
						_call,
						/*_unconditionalPayload=*/false,
						/*_includeRuntimeArgs=*/false));
				return true;
			}

			bool visit(RevertStatement const& _stmt) override
			{
				pushPayload(exportRevertPayload(
					_stmt.errorCall(),
					/*_unconditionalPayload=*/false,
					/*_includeRuntimeArgs=*/false));
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
	summary["kind"] = contract.isLibrary() ? "library" : (contract.isInterface() ? "interface" : "contract");

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
	for (auto const& [selector, functionType]: contract.interfaceFunctions())
	{
		(void) selector;
		if (!functionType)
			throw UnsupportedSolCore(
				"Foreign contract '" + contract.name()
				+ "' has an interface entry without a compiler-resolved function type.");
		methods.emplace_back(exportForeignMethodSummary(functionType));
		dispatchEntries.emplace_back(exportDispatchEntry(
			functionType, dynamic_cast<FunctionDefinition const*>(&functionType->declaration())));
	}
	summary["methods"] = methods;
	summary["dispatch_entries"] = dispatchEntries;
	return summary;
}

std::optional<Json> exportKnownExternalTarget(CompilerStack const& _compilerStack, MemberAccess const& _memberAccess)
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
	target["function"] = _memberAccess.memberName();
	target["resolutionKind"] = "cross_contract";

	auto const* functionType = dynamic_cast<FunctionType const*>(_memberAccess.annotation().type);
	if (!functionType)
		throw UnsupportedSolCore(
			"Known external target '" + _memberAccess.memberName()
			+ "' has no compiler-resolved function type.");
	FunctionAbiDescriptors abi = exportFunctionAbiDescriptors(*functionType);
	target["signature"] = functionType->externalSignature();
	target["selector"] = functionType->externalIdentifierHex();
	target["mutability"] = externalMutabilityString(functionType->stateMutability());
	target["paramsAbi"] = std::move(abi.params);
	target["returnsAbi"] = std::move(abi.returns);
	if (auto const* variableDef
		= dynamic_cast<VariableDeclaration const*>(_memberAccess.annotation().referencedDeclaration))
	{
		target["selector"] = variableDef->externalIdentifierHex();
		if (auto resolution = exportStorageGetterResolution(_compilerStack, *targetContract, *variableDef, {}))
		{
			target["resolutionKind"] = "storage_getter";
			target["resolution"] = *resolution;
		}
	}
	else if (
		auto const* functionDef
		= dynamic_cast<FunctionDefinition const*>(_memberAccess.annotation().referencedDeclaration))
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
		if (auto resolution = exportStorageGetterResolution(_compilerStack, *targetContract, *functionDef))
		{
			target["resolutionKind"] = "storage_getter";
			target["resolution"] = *resolution;
		}
		if (auto const* funType = functionDef->functionType(false))
		{
			Json dispatchEntry = exportDispatchEntry(funType, functionDef);
			if (dispatchEntry.contains("revertPayloads"))
				target["revertPayloads"] = dispatchEntry["revertPayloads"];
		}
	}
	return target;
}

bool isOptionsAwareLowLevelCall(std::string const& _memberName)
{
	return _memberName == "call" || _memberName == "delegatecall" || _memberName == "staticcall";
}

void validateLowLevelCallOptions(std::string const& _memberName, FunctionCallOptions const& _options)
{
	if (_memberName != "staticcall")
		return;
	for (auto const& name: _options.names())
		if (*name != "gas")
			throw UnsupportedSolCore(
				"Low-level `staticcall` option `" + *name
				+ "` is not modeled; only "
				  "`{gas: ...}` is supported, with the gas expression evaluated once "
				  "and its numeric value abstracted.");
}

char const* lowLevelCallKindString(std::string const& _memberName)
{
	if (_memberName == "delegatecall")
		return "delegatecall";
	if (_memberName == "staticcall")
		return "staticcall";
	if (_memberName == "call")
		return "call";
	throw UnsupportedSolCore("Unknown low-level call kind `" + _memberName + "`.");
}

Json exportTypeName(TypeName const& _typeName, bool _storage = false);
Json dataLocationEntry(VariableDeclaration const& _decl);

Json exportTypeName(TypeName const& _typeName, bool _storage)
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
			case 8:
				return Json("u8");
			case 16:
				return Json("u16");
			case 32:
				return Json("u32");
			case 64:
				return Json("u64");
			case 128:
				return Json("u128");
			case 256:
				return Json("u256");
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
			case 8:
				return Json("i8");
			case 16:
				return Json("i16");
			case 32:
				return Json("i32");
			case 64:
				return Json("i64");
			case 128:
				return Json("i128");
			case 256:
				return Json("i256");
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
			result["kind"] = _storage ? "storage_array" : "array";
			result["element"] = Json("u8");
			return result;
		}
		case Token::String:
		{
			// String type — treat as an array of u8 at EVM level
			Json result = Json::object();
			result["kind"] = _storage ? "storage_array" : "array";
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
		result["key"] = exportTypeName(mapping->keyType(), false);
		result["value"] = exportTypeName(mapping->valueType(), true);
		return result;
	}
	if (auto const* arrayType = dynamic_cast<ArrayTypeName const*>(&_typeName))
	{
		Json result = Json::object();
		Json element = exportTypeName(arrayType->baseType(), _storage);
		if (arrayType->length())
		{
			// Fixed-size array. The length is ALWAYS taken from the
			// compiler-resolved array type: the previous literal-text fast
			// path used std::stoul on the raw literal, which silently
			// truncates underscore literals (`UserPoint[1_000_000_000]`
			// parsed as size 1) — a storage-layout corruption, not an
			// error. A fixed array whose length cannot be resolved is a
			// refusal, never a silent demotion to a dynamic array (the
			// two have different storage layouts).
			auto const* resolvedArrayType = dynamic_cast<ArrayType const*>(arrayType->annotation().type);
			if (!resolvedArrayType || resolvedArrayType->isDynamicallySized())
				throw UnsupportedSolCore(
					"Fixed-size array type name has no compiler-resolved static length; "
					"refusing to guess the storage layout.");
			u256 const len = resolvedArrayType->length();
			if (len > std::numeric_limits<unsigned long>::max())
				throw UnsupportedSolCore(
					"Fixed-size array length exceeds the exporter's representable range.");
			result["kind"] = _storage ? "storage_fixed_array" : "fixed_array";
			result["element"] = element;
			result["size"] = len.convert_to<unsigned long>();
		}
		else
		{
			// Dynamic array: uint256[]
			result["kind"] = _storage ? "storage_array" : "array";
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
			return exportTypeName(*udtDef->underlyingType(), _storage);
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
		result["kind"] = _storage ? "storage_named" : "named";
		if (auto const* structDef = dynamic_cast<StructDefinition const*>(referencedDecl))
			result["name"] = exportedStructName(*structDef);
		else
			result["name"] = std::string(path.back());
		return result;
	}
	// Preserve external values as their address/selector pair and internal
	// values as a closed-table tagged value. The internal wire type carries
	// the exact parameter/return locations and mutability used to prove every
	// candidate arm type-correct; it is never reconstructed from a name.
	if (auto const* fnTypeName = dynamic_cast<FunctionTypeName const*>(&_typeName))
	{
		if (fnTypeName->visibility() == Visibility::External)
		{
			Json result = Json::object();
			result["kind"] = "external_function";
			return result;
		}
		auto const* resolved = dynamic_cast<FunctionType const*>(fnTypeName->annotation().type);
		if (!resolved || resolved->kind() != FunctionType::Kind::Internal)
			throw UnsupportedSolCore(
				"Internal function type declaration has no matching compiler-resolved FunctionType.");
		Json result = Json::object();
		result["kind"] = "internal_function";
		result["params"] = Json::array();
		for (auto const& param: fnTypeName->parameterTypes())
		{
			Json component = Json::object();
			Json location = dataLocationEntry(*param);
			bool storage = location.is_string() && location.get<std::string>() == "storage";
			component["type"] = exportTypeName(param->typeName(), storage);
			component["location"] = location.is_null() ? Json("none") : location;
			result["params"].emplace_back(std::move(component));
		}
		result["returns"] = Json::array();
		for (auto const& ret: fnTypeName->returnParameterTypes())
		{
			Json component = Json::object();
			Json location = dataLocationEntry(*ret);
			bool storage = location.is_string() && location.get<std::string>() == "storage";
			component["type"] = exportTypeName(ret->typeName(), storage);
			component["location"] = location.is_null() ? Json("none") : location;
			result["returns"].emplace_back(std::move(component));
		}
		switch (fnTypeName->stateMutability())
		{
		case StateMutability::Pure:
			result["mutability"] = "pure";
			break;
		case StateMutability::View:
			result["mutability"] = "view";
			break;
		case StateMutability::Payable:
			result["mutability"] = "payable";
			break;
		default:
			result["mutability"] = "nonpayable";
			break;
		}
		(void) registerInternalFnTable(*resolved, result);
		return result;
	}
	throw UnsupportedSolCore("Unsupported type node in SolCore exporter.");
}

/// Create a producer-typed default expression for a declared type.
/// Consumers recursively materialize the default; the exporter never
/// substitutes a scalar zero for an aggregate or first-class value.
Json defaultValueForTypeName(TypeName const& _typeName, bool _storage = false)
{
	Json result = Json::object();
	result["kind"] = "typed_default";
	result["type"] = exportTypeName(_typeName, _storage);
	return result;
}

/// Wire name for a RESOLVED data location of a reference-typed
/// parameter/return ("storage" / "memory" / "calldata"). "transient" is not a
/// legal parameter/return location; emitted honestly so the consumer's
/// fail-closed unknown-location path refuses it instead of misreading it as a
/// value copy. Shared by `exportParam`'s `location` field and
/// `exportFunction`'s `returnLocations` field so the two halves of the wire
/// convention can never drift.
char const* dataLocationName(DataLocation _loc)
{
	switch (_loc)
	{
	case DataLocation::Storage:
		return "storage";
	case DataLocation::Memory:
		return "memory";
	case DataLocation::CallData:
		return "calldata";
	case DataLocation::Transient:
		return "transient";
	}
	return "memory";
}

/// The wire data-location entry for a declared parameter/return variable:
/// a location string for reference-typed (and mapping-typed) declarations,
/// JSON null for value-typed ones (which have no data location). Mapping
/// types are not ReferenceType in solc's hierarchy but are storage-only by
/// language rule; emitted explicitly so they do not read as location-less.
Json dataLocationEntry(VariableDeclaration const& _decl)
{
	if (auto const* type = _decl.annotation().type)
	{
		if (auto const* refType = dynamic_cast<ReferenceType const*>(type))
			return Json(dataLocationName(refType->location()));
		if (type->category() == Type::Category::Mapping)
			return Json("storage");
	}
	return Json();
}

Json exportParam(
	VariableDeclaration const& _decl,
	bool _abiVisible = false,
	bool _forLibrary = false)
{
	Json result = Json::object();
	result["sourceDeclarationId"] = std::to_string(_decl.id());
	result["name"] = _decl.name().empty() ? ("arg" + std::to_string(stableSyntheticNodeId(_decl))) : _decl.name();
	Json location = dataLocationEntry(_decl);
	bool storage = location.is_string() && location.get<std::string>() == "storage";
	result["type"] = exportTypeName(_decl.typeName(), storage);
	if (!location.is_null())
		result["location"] = std::move(location);
	if (_abiVisible)
		result["abi"] = exportAbiDescriptor(_decl.name(), _decl.annotation().type, _forLibrary);
	return result;
}

/// Canonical SolCore carrier type seen by an inline-assembly capture. Ordinary
/// parameter/local rows carry memory as a separate `location` field and the
/// OCaml parser qualifies their type. Assembly-interface rows are closed and
/// have no location field, so apply that qualification at the producer.
Json exportAssemblyCaptureType(VariableDeclaration const& _decl)
{
	Json location = dataLocationEntry(_decl);
	bool storage = location.is_string() && location.get<std::string>() == "storage";
	Json result = exportTypeName(_decl.typeName(), storage);
	if (location.is_string() && location.get<std::string>() == "memory")
	{
		Json memoryRef = Json::object();
		memoryRef["kind"] = "memory_ref";
		memoryRef["referent"] = std::move(result);
		return memoryRef;
	}
	return result;
}
Json exportConstructorParamsAbi(FunctionDefinition const& _constructor)
{
	if (!_constructor.isConstructor())
		throw UnsupportedSolCore("Constructor ABI requested for a non-constructor declaration.");
	Json result = Json::array();
	for (auto const& parameter: _constructor.parameters())
		result.emplace_back(exportAbiDescriptor(parameter->name(), parameter->annotation().type, false));
	return result;
}

bool inheritedConstructorHasWork(ContractDefinition const& _contract)
{
	for (ContractDefinition const* base: _contract.annotation().linearizedBaseContracts)
	{
		if (base == &_contract)
			continue;
		for (VariableDeclaration const* stateVariable: base->stateVariables())
			if (!stateVariable->isConstant() && stateVariable->value())
				return true;
		FunctionDefinition const* baseConstructor = base->constructor();
		if (!baseConstructor)
			continue;
		if (
			(baseConstructor->isImplemented() && !baseConstructor->body().statements().empty())
			|| _contract.annotation().baseConstructorArguments.count(baseConstructor) != 0
		)
			return true;
	}
	return false;
}

Json exportConstructorDisposition(ContractDefinition const& _contract)
{
	Json result = Json::object();
	FunctionDefinition const* constructor = _contract.constructor();
	if (!constructor)
	{
		if (inheritedConstructorHasWork(_contract))
		{
			result["kind"] = "exported";
			result["declarationId"] = std::to_string(_contract.id());
			result["function"] = "constructor";
			result["paramsAbi"] = Json::array();
			result["sourceLocation"] = sourceLocation(_contract.location());
		}
		else
			result["kind"] = "implicit";
		return result;
	}
	if (!constructor->isImplemented())
		throw UnsupportedSolCore("An explicit constructor has no exportable body.");
	result["kind"] = "exported";
	result["declarationId"] = std::to_string(constructor->id());
	result["function"] = "constructor";
	result["paramsAbi"] = exportConstructorParamsAbi(*constructor);
	result["sourceLocation"] = sourceLocation(constructor->location());
	return result;
}

Json immutableGet(VariableDeclaration const& _declaration)
{
	if (!_declaration.isStateVariable() || !_declaration.immutable())
		throw UnsupportedSolCore("Immutable read requested for a non-immutable declaration.");
	Json result = Json::object();
	result["kind"] = "immutable_get";
	result["declarationId"] = std::to_string(_declaration.id());
	result["name"] = _declaration.name();
	return result;
}

Json immutableSet(VariableDeclaration const& _declaration, Json _value)
{
	if (!_declaration.isStateVariable() || !_declaration.immutable())
		throw UnsupportedSolCore("Immutable write requested for a non-immutable declaration.");
	Json result = Json::object();
	result["kind"] = "immutable_set";
	result["declarationId"] = std::to_string(_declaration.id());
	result["name"] = _declaration.name();
	result["value"] = std::move(_value);
	return result;
}

Json exportField(VariableDeclaration const& _decl)
{
	Json result = Json::object();
	result["name"] = _decl.name();
	result["type"] = exportTypeName(_decl.typeName());
	return result;
}

/// Struct MEMBER export for `type_decls`: `exportField` plus three additive
/// placement facts from solc's own layout, never re-derived downstream:
///   - `storageSlot` — the member's slot offset within the struct's OWN
///     storage frame (relative to the struct's first slot), from
///     `StructType::storageOffsetsOfMember`. Same "slot within the parent
///     frame" meaning as the `storageSlot` on contract storage fields.
///     Emitted for EVERY laid-out member, packed or not; consumers needing a
///     slot-addressable member must additionally require `byteOffset == 0`
///     (load-bearing for the E4a library slot-word convention, whose
///     reference-typed members always own whole slots and so always carry
///     offset 0).
///   - `byteOffset` — the member's intra-slot byte offset, low end first
///     (`storageOffsetsOfMember`'s second component).
///   - `byteWidth` — the bytes the member occupies within its slot
///     (`Type::storageBytes`; 32 for full-slot and reference-typed members).
/// All three are omitted together when solc has no layout for the member;
/// consumers fail closed on absence.
Json exportStructMemberField(StructDefinition const& _structDef, VariableDeclaration const& _member)
{
	Json result = exportField(_member);
	try
	{
		if (auto const* structType = TypeProvider::structType(_structDef, DataLocation::Storage))
		{
			auto const& offsets = structType->storageOffsetsOfMember(_member.name());
			result["storageSlot"] = offsets.first.str();
			result["byteOffset"] = static_cast<int>(offsets.second);
			if (auto const* memberType = structType->members(nullptr).memberType(_member.name()))
				result["byteWidth"] = static_cast<int>(memberType->storageBytes());
		}
	}
	catch (...)
	{
		// Layout unavailable (error type, unnameable member, ...): omit the
		// fields; consumers fail closed on absence.
	}
	return result;
}

Json exportStorageField(
	CompilerStack const& _compilerStack, ContractDefinition const& _contract, VariableDeclaration const& _decl)
{
	Json result = Json::object();
	result["name"] = _decl.name();
	result["type"] = exportTypeName(_decl.typeName(), true);
	if (auto slot = lookupStorageSlot(_compilerStack, _contract, _decl.name()))
		result["storageSlot"] = *slot;
	return result;
}

Json exportEventParam(VariableDeclaration const& _decl)
{
	Json result = Json::object();
	result["name"] = _decl.name().empty() ? ("arg" + std::to_string(stableSyntheticNodeId(_decl))) : _decl.name();
	result["type"] = exportTypeName(_decl.typeName());
	result["indexed"] = _decl.isIndexed();
	result["abi"] = exportAbiDescriptor(_decl.name(), _decl.annotation().type, false);
	return result;
}

/// Defined below, next to `assignExportedEventNames` (which owns the memo they
/// read); forward-declared here because `exportEvent` precedes them in the file.
std::string exportedEventName(EventDefinition const& _event);
std::string eventSoliditySignature(EventDefinition const& _event);

std::string abiSignatureComponent(Json const& _descriptor)
{
	if (!_descriptor.is_object() || !_descriptor.contains("type") || !_descriptor["type"].is_string()
		|| !_descriptor.contains("components") || !_descriptor["components"].is_array())
		throw UnsupportedSolCore("Malformed ABI descriptor while constructing an event signature.");
	std::string type = _descriptor["type"].get<std::string>();
	if (type.rfind("tuple", 0) != 0)
		return type;
	std::string signature = "(";
	bool first = true;
	for (Json const& component: _descriptor["components"])
	{
		if (!first)
			signature += ",";
		first = false;
		signature += abiSignatureComponent(component);
	}
	return signature + ")" + type.substr(5);
}

Json exportEvent(EventDefinition const& _event)
{
	Json result = Json::object();
	result["name"] = exportedEventName(_event);
	result["declarationId"] = std::to_string(static_cast<int64_t>(_event.id()));
	result["sourceLocation"] = sourceLocation(_event.location());
	result["anonymous"] = _event.isAnonymous();
	result["params"] = Json::array();
	std::string descriptorSignature = _event.name() + "(";
	bool first = true;
	for (auto const& parameter: _event.parameters())
	{
		Json param = exportEventParam(*parameter);
		if (!first)
			descriptorSignature += ",";
		first = false;
		descriptorSignature += abiSignatureComponent(param["abi"]);
		result["params"].emplace_back(std::move(param));
	}
	descriptorSignature += ")";
	std::string compilerSignature = eventSoliditySignature(_event);
	if (compilerSignature.empty() || compilerSignature != descriptorSignature)
		throw UnsupportedSolCore(
			"Event '" + _event.name()
			+ "' has inconsistent compiler-resolved signature and ABI descriptors.");
	result["signature"] = std::move(descriptorSignature);
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
		if (_member == "sig")
			return "msgSig";
	}
	// Invariant: transaction origin is producer-resolved environment data.
	// The exporter owns the exact `tx.origin` identity and emits the same
	// first-class state_get shape as `msg.sender`; consumers never infer it
	// from a declaration name or constrain it to the immediate caller.
	if (_base == "tx" && _member == "origin")
		return "origin";
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
Json exportHoistedUnaryMutation(UnaryOperation const& _unary);
Json exportHoistedAssignExpr(Assignment const& _assignment);
std::optional<Json> exportStorageRefValue(Expression const& _expr);
bool hasResidualStorageRefRoot(Expression const& _expr);
bool isStorageReferenceType(Type const* _type);
Json storageRefWireType(Type const* _referentType);
std::optional<StorageRefTarget> resolveStorageRefCallRoot(
	FunctionCall const& _call,
	ASTNode const& _snapshotOwner,
	std::vector<Json>& _snapshots);


/// Static selector node for `abi.encodeCall`'s first argument when it names a
/// FUNCTION DECLARATION rather than a function-pointer VALUE.
///
/// `abi.encodeCall(f, (args))` consumes only `f`'s 4-byte selector — the
/// address half of an external function pointer is never encoded. In the
/// NAMESPACE forms (`I.echo`, `C.echo`, `L.increment`) there is no receiver at
/// all: solc types them `FunctionType::Kind::Declaration`, so the
/// external-function-VALUE arm in exportExpr does not fire and the base (a
/// bare contract/interface/library identifier) has no address representation,
/// which used to fail the whole function closed. Emitting the statically
/// resolved selector constant instead is exactly the datum encodeCall needs
/// and invents nothing: both the selector and the signature come from the
/// declaration solc already bound.
///
/// Returns nullopt for `Kind::External` (a bound member or a function-typed
/// VALUE — `I(target).echo`, a storage/local/try-bound function pointer),
/// which keeps those on the existing extfn_pack path and leaves the
/// fn-pointer value model's territory untouched. Also nullopt when nothing
/// resolves, so the caller falls back to the ordinary argument export and
/// whatever refusal it produces.
std::optional<Json> exportDeclaredCalleeSelector(Expression const& _callee)
{
	auto const* funType = dynamic_cast<FunctionType const*>(_callee.annotation().type);
	if (!funType || funType->kind() == FunctionType::Kind::External)
		return std::nullopt;

	Declaration const* referenced = nullptr;
	std::string methodName;
	std::string contractId;
	if (auto const* member = dynamic_cast<MemberAccess const*>(&_callee))
	{
		referenced = member->annotation().referencedDeclaration;
		methodName = member->memberName();
		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&member->expression()))
			if (auto const* cd = dynamic_cast<ContractDefinition const*>(baseIdent->annotation().referencedDeclaration))
				contractId = cd->name();
	}
	else if (auto const* ident = dynamic_cast<Identifier const*>(&_callee))
	{
		referenced = ident->annotation().referencedDeclaration;
		methodName = ident->name();
	}
	auto const* funcDef = dynamic_cast<FunctionDefinition const*>(referenced);
	if (!funcDef)
		return std::nullopt;

	// Prefer the annotation FunctionType (already carries the declaration for
	// a qualified member access); fall back to synthesizing the external
	// interface type from the FunctionDefinition, exactly as the `.selector`
	// arm in exportExpr does.
	std::string selectorHex;
	std::string signature;
	// externalSignature()/externalIdentifierHex() assert for a declaration
	// with no external interface type (e.g. an internal function with a
	// storage-reference parameter). solc's type checker does not admit such a
	// callee for abi.encodeCall, but resolve defensively — the same try/catch
	// discipline exportOrigins already uses around externalSignature() — so an
	// unexpected shape declines this arm instead of aborting the export.
	try
	{
		if (funType->hasDeclaration())
		{
			selectorHex = funType->externalIdentifierHex();
			signature = funType->externalSignature();
		}
		else
		{
			selectorHex = funcDef->externalIdentifierHex();
			FunctionType ft(*funcDef);
			if (FunctionType const* iface = ft.interfaceFunctionType())
				signature = iface->externalSignature();
		}
	}
	catch (...)
	{
		return std::nullopt;
	}
	if (selectorHex.empty() || signature.empty())
		return std::nullopt;

	Json result = Json::object();
	result["kind"] = "method_selector";
	result["method_name"] = methodName;
	result["method_signature"] = signature;
	result["selector_hex"] = selectorHex;
	if (contractId.empty())
		result["contractId"] = nullptr;
	else
		result["contractId"] = contractId;
	return result;
}

Json u256Literal(std::string const& _value)
{
	Json result = Json::object();
	result["kind"] = "u256";
	result["value"] = _value;
	return result;
}

Json emptyBytesLiteral()
{
	Json target = Json::object();
	target["kind"] = "bytes";
	Json result = Json::object();
	result["kind"] = "typed_literal";
	result["encoding"] = "hex";
	result["sourceBytes"] = "";
	result["width"] = 0;
	result["target"] = std::move(target);
	return result;
}

Json localExpr(std::string const& _name)
{
	Json result = Json::object();
	result["kind"] = "local";
	result["name"] = _name;
	return result;
}
/// Conservative detector for expressions whose evaluation order can be
/// observed. Besides explicit writes, a non-conversion call may itself mutate
/// or observe state. This predicate only decides whether to materialize an
/// already measured order; it never proves purity.
struct EvalOrderRelevantScanner: ASTConstVisitor
{
	bool relevant = false;

	bool visit(Assignment const&) override
	{
		relevant = true;
		return false;
	}

	bool visit(UnaryOperation const& _unary) override
	{
		if (_unary.getOperator() == Token::Inc || _unary.getOperator() == Token::Dec)
		{
			relevant = true;
			return false;
		}
		return true;
	}

	bool visit(FunctionCall const& _call) override
	{
		// Dynamic array allocation itself neither observes nor mutates contract
		// state. Keep visiting its length expression (which may still contain
		// an order-relevant call) without classifying `new T[](n)` as a call
		// effect; this preserves typed conditional nodes such as
		// `flag ? new bytes(0) : hex"ab"`.
		if (dynamic_cast<NewExpression const*>(&_call.expression()))
			return true;
		if (!_call.annotation().kind.set() || *_call.annotation().kind != FunctionCallKind::TypeConversion)
		{
			relevant = true;
			return false;
		}
		return true;
	}
};

bool evalOrderRelevant(Expression const& _expr)
{
	EvalOrderRelevantScanner scanner;
	_expr.accept(scanner);
	return scanner.relevant;
}

Json blockFromStatements(std::vector<Json> _statements)
{
	Json block = Json::object();
	block["kind"] = "block";
	block["statements"] = Json::array();
	for (Json& statement: _statements)
		block["statements"].emplace_back(std::move(statement));
	return block;
}

Json localAssignment(std::string const& _name, Json _value)
{
	Json statement = Json::object();
	statement["kind"] = "assign";
	statement["name"] = _name;
	statement["value"] = std::move(_value);
	return statement;
}

/// Evaluate one expression in an isolated, once-evaluated scope and append
/// both its nested mutation lowering and a capture-let to the parent scope.
/// Isolation is intentional: the caller has selected a measured source order,
/// so sibling read/write conflicts are no longer "unspecified order".
/// `_contextualType`, when present, is the type checker's resolved destination
/// type for this value (for example an internal-call parameter). It is
/// authoritative over the expression's pre-conversion literal type.
/// `_storageRefValue` retains the producer-resolved reference itself instead of
/// dereferencing it; assignment-lvalue reconstruction consumes that exact
/// identity after the once-evaluated call root has been memoized.
Json pinExpressionOnce(
	Expression const& _expr,
	std::string const& _label,
	Type const* _contextualType = nullptr,
	bool _storageRefValue = false)
{
	HoistScope* parent = activeHoistScope;
	if (!parent || !parent->allowed)
		throw UnsupportedSolCore(
			"Measured evaluation-order normalization requires a once-evaluated "
			"statement context.");

	auto memoIt = parent->memo.find(static_cast<int64_t>(_expr.id()));
	if (memoIt != parent->memo.end())
		return localExpr(memoIt->second);

	bool storageRefValue = _storageRefValue;
	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
	{
		std::vector<Json> probeSnapshots;
		auto probe = resolveStorageRefCallRoot(*call, *call, probeSnapshots);
		storageRefValue = storageRefValue || (probe && probe->snapshotAsTypedCall);
	}

	Json value;
	std::vector<Json> nested;
	{
		HoistScopeGuard childScope({&_expr});
		if (storageRefValue)
		{
			auto reference = exportStorageRefValue(_expr);
			if (!reference)
				throw UnsupportedSolCore(
					"Measured assignment-lvalue storage reference did not resolve to an exact typed identity.");
			value = std::move(*reference);
		}
		else
			value = exportExpr(_expr);
		nested = childScope.takeStatements();
	}
	for (Json& statement: nested)
		parent->statements.emplace_back(std::move(statement));

	Type const* type = _contextualType ? _contextualType : _expr.annotation().type;
	if (!type)
		throw UnsupportedSolCore(
			"Measured evaluation-order normalization requires an annotated "
			"or contextually resolved expression type.");
	std::string tempName = "__solcore_evalorder_" + _label + "_" + std::to_string(stableSyntheticNodeId(_expr));
	Json letStmt = Json::object();
	letStmt["kind"] = "let";
	letStmt["sourceDeclarationId"] = Json();
	letStmt["name"] = tempName;
	letStmt["type"] = storageRefValue ? storageRefWireType(type) : exportResolvedType(type);
	letStmt["value"] = std::move(value);
	parent->statements.emplace_back(std::move(letStmt));
	parent->memo.emplace(static_cast<int64_t>(_expr.id()), tempName);
	return localExpr(tempName);
}
/// A literal has no run-time observation to order, and its annotation may be
/// a contextual literal type that must not be invented as a temporary type.
/// Every other child of a selected order row is captured through the ordinary
/// typed pinning path.
Json captureMeasuredOrderChild(Expression const& _expr, std::string const& _label)
{
	if (dynamic_cast<Literal const*>(&_expr))
		return exportExpr(_expr);
	return pinExpressionOnce(_expr, _label);
}

/// Materialize the executable children of an assignment lvalue in the order
/// used by the compiler's legacy Assignment/IndexAccess visitors: after the
/// RHS, tuple components are visited left-to-right and an index visits its
/// base before its key. Member selection itself is pure; only its base needs
/// evaluation. Re-export during reconstruction consumes the parent memo.
void pinAssignmentLValueChildren(Expression const& _lvalue, std::string const& _label)
{
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_lvalue))
	{
		if (!tuple->isInlineArray())
			for (size_t i = 0; i < tuple->components().size(); ++i)
				if (tuple->components()[i])
					pinAssignmentLValueChildren(
						*tuple->components()[i], _label + "_tuple_" + std::to_string(i));
		return;
	}
	if (auto const* index = dynamic_cast<IndexAccess const*>(&_lvalue))
	{
		pinAssignmentLValueChildren(index->baseExpression(), _label + "_base");
		if (index->indexExpression())
			(void) captureMeasuredOrderChild(*index->indexExpression(), _label + "_index");
		return;
	}
	if (auto const* member = dynamic_cast<MemberAccess const*>(&_lvalue))
	{
		pinAssignmentLValueChildren(member->expression(), _label + "_base");
		return;
	}
	if (evalOrderRelevant(_lvalue))
	{
		// The compiler-resolved storage location is authoritative. Calls such
		// as `getStringSlot(store)` may not yet be registered as residual roots
		// at this early LHS snapshot point, but their annotated result is still
		// a first-class storage reference and must take the typed-reference pin.
		if (isStorageReferenceType(_lvalue.annotation().type)
			|| hasResidualStorageRefRoot(_lvalue))
			(void) pinExpressionOnce(_lvalue, _label, nullptr, /*_storageRefValue=*/true);
		else
			(void) captureMeasuredOrderChild(_lvalue, _label);
	}
}


struct CallEvaluation
{
	Json target;
	std::vector<Json> options;
	std::vector<Json> arguments;
};

/// Materialize the measured external-call sequence: target first, call
/// options in their written order, arguments last. Solc preserves
/// FunctionCallOptions::names/options in source order. Every participant is
/// captured when any participant is order-observable, so no expression can be
/// duplicated by later field-specific lowering.
CallEvaluation
exportCallEvaluation(
	FunctionCall const& _call,
	Expression const& _target,
	FunctionCallOptions const* _options,
	FunctionType const* _calleeType = nullptr)
{
	bool observable = evalOrderRelevant(_target);
	if (_options)
		for (auto const& option: _options->options())
			observable = observable || evalOrderRelevant(*option);
	for (auto const& argument: _call.arguments())
		observable = observable || evalOrderRelevant(*argument);

	// Destination parameter type per SOURCE-order argument, from the callee's
	// compiler-resolved FunctionType. A pinned literal argument must carry the
	// DECLARED parameter type: its own annotation is a rational constant with
	// no wire representation, and deriving a mobile type here would re-derive
	// what the producer already resolved. Positional calls map by index; named
	// calls map through the callee's parameter names. Any shape this cannot
	// map exactly (arity mismatch, unknown name) keeps the typeless pin, whose
	// rational-type export stays fail-closed.
	auto destinationType = [&](size_t sourceIndex) -> Type const* {
		if (!_calleeType)
			return nullptr;
		auto const& parameterTypes = _calleeType->parameterTypes();
		if (_call.names().empty())
			return sourceIndex < parameterTypes.size() ? parameterTypes[sourceIndex] : nullptr;
		auto const& parameterNames = _calleeType->parameterNames();
		if (sourceIndex >= _call.names().size() || parameterNames.size() != parameterTypes.size())
			return nullptr;
		auto const& argumentName = *_call.names()[sourceIndex];
		for (size_t parameterIndex = 0; parameterIndex < parameterNames.size(); ++parameterIndex)
			if (parameterNames[parameterIndex] == argumentName)
				return parameterTypes[parameterIndex];
		return nullptr;
	};

	CallEvaluation result;
	result.target = observable ? pinExpressionOnce(_target, "call_target") : exportExpr(_target);
	if (_options)
		for (size_t i = 0; i < _options->options().size(); ++i)
		{
			std::string label = "call_option_" + *_options->names()[i];
			result.options.emplace_back(
				observable ? pinExpressionOnce(*_options->options()[i], label) : exportExpr(*_options->options()[i]));
		}
	for (size_t i = 0; i < _call.arguments().size(); ++i)
		result.arguments.emplace_back(
			observable ? pinExpressionOnce(*_call.arguments()[i], "call_arg_" + std::to_string(i), destinationType(i))
					   : exportExpr(*_call.arguments()[i]));
	return result;
}

/// Conservative side-effect-freedom for CALLER argument expressions consumed
/// by the [E1] call-root resolution below: an argument may be re-evaluated at
/// a different position (a key snapshot `let`), or NOT evaluated at all (an
/// argument the callee's returned path never touches), so anything that can
/// write, revert-with-effect, or observe evaluation order must refuse. Reads
/// (identifier / member / index chains) and literals are safe: the whole
/// resolution happens within one statement with no interleaved writes.
bool isSideEffectFreeExpr(Expression const& _expr)
{
	if (dynamic_cast<Literal const*>(&_expr))
		return true;
	if (dynamic_cast<Identifier const*>(&_expr))
		return true;
	if (auto const* member = dynamic_cast<MemberAccess const*>(&_expr))
		return isSideEffectFreeExpr(member->expression());
	if (auto const* index = dynamic_cast<IndexAccess const*>(&_expr))
		return index->indexExpression() && isSideEffectFreeExpr(index->baseExpression())
			   && isSideEffectFreeExpr(*index->indexExpression());
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expr))
		return !tuple->isInlineArray() && tuple->components().size() == 1 && tuple->components().front()
			   && isSideEffectFreeExpr(*tuple->components().front());
	return false;
}

std::optional<StorageRefTarget>
resolveStorageRefPathRec(Expression const& _expr, ASTNode const& _snapshotOwner, std::vector<Json>& _snapshots);

/// [E1 — call-return storage-ref path propagation] Resolve the RETURNED PATH
/// of `_callee` (whose body is exactly `return <path-expr>;`) as a
/// caller-context StorageRefTarget, with the callee's parameters bound to the
/// caller's argument expressions:
///   - a state-variable root composes directly (an internal callee always
///     shares the caller's storage space: contract/base functions read their
///     own — inherited — layout, and libraries cannot declare state
///     variables);
///   - a parameter root substitutes the caller's argument expression,
///     resolved recursively in CALLER context (state vars, tracked aliases
///     and nested resolvable calls all compose);
///   - an index/mapping key must be a literal (context-free) or a bare
///     parameter identifier (substituted by the caller's argument
///     expression, which the caller guarantees side-effect-free), snapshot
///     into a bind-site `let` exactly like a direct bind's key.
/// Anything else — arithmetic over keys, nested calls inside the callee,
/// conditional paths — fails closed (nullopt), keeping the copy-`let`
/// fallback and its E0 write-through guard.
static thread_local bool suppressStorageRefCallResolution = false;

struct StorageRefCallResolutionGuard
{
	StorageRefCallResolutionGuard(): m_previous(suppressStorageRefCallResolution)
	{
		suppressStorageRefCallResolution = true;
	}
	~StorageRefCallResolutionGuard() { suppressStorageRefCallResolution = m_previous; }
	StorageRefCallResolutionGuard(StorageRefCallResolutionGuard const&) = delete;
	StorageRefCallResolutionGuard& operator=(StorageRefCallResolutionGuard const&) = delete;

private:
	bool m_previous;
};

void appendNewStorageRefSnapshots(std::vector<Json>& _destination, std::vector<Json>&& _candidate, size_t _prefixSize)
{
	for (size_t i = _prefixSize; i < _candidate.size(); ++i)
		_destination.emplace_back(std::move(_candidate[i]));
}
/// Export one dynamic storage-path component without leaking its nested
/// hoists into the enclosing statement ahead of an earlier path component.
/// The caller appends components root-outward; splice this component's own
/// hoists at that exact position, then snapshot its value. If a measured-order
/// parent already pinned the whole expression, reuse that typed capture.
Json exportStorageRefSnapshotValue(Expression const& _expr, std::vector<Json>& _snapshots)
{
	if (activeHoistScope)
	{
		auto memoIt = activeHoistScope->memo.find(static_cast<int64_t>(_expr.id()));
		if (memoIt != activeHoistScope->memo.end())
			return localExpr(memoIt->second);
	}

	Json value;
	std::vector<Json> nestedStatements;
	{
		HoistScopeGuard componentScope({&_expr});
		value = exportExpr(_expr);
		nestedStatements = componentScope.takeStatements();
	}
	for (Json& statement: nestedStatements)
		_snapshots.emplace_back(std::move(statement));

	return value;
}
/// Resolve the closed assembly storage-pointer-return shapes:
///
///     function getXSlot(bytes32 slot) internal pure
///         returns (XSlot storage result)
///     { assembly { result.slot := slot } }
///
/// and the storage-pointer identity used by ShortStrings:
///
///     function fallback(string storage store)
///         returns (string storage result)
///     { assembly { result.slot := store.slot } }
///
/// Invariant: the exporter owns slot-word provenance.  Admission requires the
/// RHS to be one exact compiler external reference, optionally copied through
/// Yul identifier-only moves.  A bare value parameter is snapshotted as a raw
/// slot word; a captured storage parameter's `.slot` carries the caller's
/// structurally resolved storage path. Arithmetic, calls, literals, unknown
/// locals, mixed suffixes, and every other statement shape fail closed.
std::optional<StorageRefTarget> resolveAssemblyRawSlotReturnPath(
	FunctionDefinition const& _callee,
	std::map<VariableDeclaration const*, Expression const*> const& _paramBinding,
	ASTNode const& _snapshotOwner,
	std::vector<Json>& _snapshots)
{
	if (!_callee.isImplemented() || _callee.returnParameters().size() != 1)
		return std::nullopt;
	VariableDeclaration const* returnSlot = _callee.returnParameters().front().get();
	if (!returnSlot || returnSlot->referenceLocation() != VariableDeclaration::Location::Storage)
		return std::nullopt;
	Block const& body = _callee.body();
	if (body.statements().size() != 1)
		return std::nullopt;
	auto const* assembly = dynamic_cast<InlineAssembly const*>(body.statements().front().get());
	if (!assembly)
		return std::nullopt;

	auto const& references = assembly->annotation().externalReferences;
	using ExternalIdentifierInfo = InlineAssemblyAnnotation::ExternalIdentifierInfo;
	std::map<std::string, ExternalIdentifierInfo const*> moveOrigins;
	auto exactOrigin = [&](yul::Identifier const& _identifier) -> ExternalIdentifierInfo const*
	{
		auto external = references.find(&_identifier);
		if (external != references.end())
			return &external->second;
		auto local = moveOrigins.find(_identifier.name.str());
		return local == moveOrigins.end() ? nullptr : local->second;
	};

	ExternalIdentifierInfo const* rhsOrigin = nullptr;
	bool sawReturnAssignment = false;
	bool sawNonMove = false;
	yul::Block const& root = assembly->operations().root();
	for (yul::Statement const& statement: root.statements)
	{
		if (auto const* declaration = std::get_if<yul::VariableDeclaration>(&statement))
		{
			if (declaration->variables.size() != 1)
			{
				sawNonMove = true;
				continue;
			}
			std::string const name = declaration->variables.front().name.str();
			auto const* rhs = declaration->value
				? std::get_if<yul::Identifier>(declaration->value.get())
				: nullptr;
			auto const* origin = rhs ? exactOrigin(*rhs) : nullptr;
			if (origin)
				moveOrigins[name] = origin;
			else
				moveOrigins.erase(name);
			if (declaration->value && !origin)
				sawNonMove = true;
			continue;
		}

		auto const* assignment = std::get_if<yul::Assignment>(&statement);
		if (!assignment || assignment->variableNames.size() != 1)
		{
			sawNonMove = true;
			continue;
		}
		yul::Identifier const& lhs = assignment->variableNames.front();
		auto lhsReference = references.find(&lhs);
		bool const assignsReturnSlot =
			lhsReference != references.end()
			&& lhsReference->second.declaration == returnSlot
			&& lhsReference->second.suffix == "slot";
		auto const* rhs = assignment->value
			? std::get_if<yul::Identifier>(assignment->value.get())
			: nullptr;
		if (assignsReturnSlot)
		{
			if (sawReturnAssignment || !rhs)
				throw UnsupportedSolCore(
					"Assembly-assigned storage-pointer return `" + returnSlot->name()
					+ ".slot` has a slot expression that is not a single compiler-owned Solidity value; "
					  "raw-slot provenance is unresolvable.");
			sawReturnAssignment = true;
			rhsOrigin = exactOrigin(*rhs);
			if (!rhsOrigin)
				throw UnsupportedSolCore(
					"Assembly-assigned storage-pointer return `" + returnSlot->name()
					+ ".slot` has no exact compiler-owned slot-word provenance.");
			continue;
		}

		if (lhsReference != references.end())
		{
			sawNonMove = true;
			continue;
		}
		auto const* origin = rhs ? exactOrigin(*rhs) : nullptr;
		if (origin)
			moveOrigins[lhs.name.str()] = origin;
		else
			moveOrigins.erase(lhs.name.str());
		if (!origin)
			sawNonMove = true;
	}

	if (!sawReturnAssignment)
		return std::nullopt;
	if (sawNonMove || !rhsOrigin)
		throw UnsupportedSolCore(
			"Assembly-assigned storage-pointer return `" + returnSlot->name()
			+ ".slot` has no exact compiler-owned slot-word provenance.");

	auto const* slotParameter
		= dynamic_cast<VariableDeclaration const*>(rhsOrigin->declaration);
	auto argument = slotParameter ? _paramBinding.find(slotParameter) : _paramBinding.end();
	if (argument == _paramBinding.end() || !argument->second)
		throw UnsupportedSolCore(
			"Assembly-assigned storage-pointer return `" + returnSlot->name()
			+ ".slot` depends on a value that is not an exactly bound helper parameter; "
			  "raw-slot provenance is unresolvable.");

	if (rhsOrigin->suffix == "slot")
	{
		if (!isStorageRefParameter(slotParameter))
			throw UnsupportedSolCore(
				"Assembly-assigned storage-pointer return `" + returnSlot->name()
				+ ".slot` has no exact compiler-owned slot-word provenance.");
		auto target = resolveStorageRefPathRec(*argument->second, _snapshotOwner, _snapshots);
		if (!target)
			throw UnsupportedSolCore(
				"Assembly-assigned storage-pointer return `" + returnSlot->name()
				+ ".slot` depends on a captured storage slot whose caller path is unresolved.");
		// A `.slot` return may reinterpret the caller root as a different
		// storage referent (ShortStrings' string -> StringSlot). Pinning the
		// resolved root as the callee return type would create an ill-typed
		// alias; retain the pure helper call as the typed snapshot value.
		target->snapshotAsTypedCall
			= slotParameter->annotation().type != returnSlot->annotation().type;
		return target;
	}
	if (!rhsOrigin->suffix.empty())
		throw UnsupportedSolCore(
			"Assembly-assigned storage-pointer return `" + returnSlot->name()
			+ ".slot` has no exact compiler-owned slot-word provenance.");

	std::string tempName = storageRefKeyTempName(_snapshotOwner, _snapshots.size());
	Json letStmt = Json::object();
	letStmt["kind"] = "let";
	letStmt["sourceDeclarationId"] = Json();
	letStmt["name"] = tempName;
	Type const* slotType = argument->second->annotation().type;
	auto simpleSlotType = slotType ? exportSimpleType(*slotType) : std::nullopt;
	letStmt["type"] = simpleSlotType.has_value() ? *simpleSlotType : Json("u256");
	letStmt["value"] = exportStorageRefSnapshotValue(*argument->second, _snapshots);
	_snapshots.emplace_back(std::move(letStmt));

	StorageRefTarget target;
	target.rootKind = StorageRefTarget::RootKind::RawSlot;
	target.rootType = returnSlot->annotation().type;
	target.rawSlot = localExpr(tempName);
	return target;
}

std::optional<StorageRefTarget> resolveCalleeReturnPath(
	Expression const& _expr,
	std::map<VariableDeclaration const*, Expression const*> const& _paramBinding,
	ASTNode const& _snapshotOwner,
	std::vector<Json>& _snapshots);

std::optional<StorageRefTarget> resolveSameCalleeStorageRefBranches(
	Expression const& _trueExpr,
	Expression const& _falseExpr,
	std::map<VariableDeclaration const*, Expression const*> const& _paramBinding,
	ASTNode const& _snapshotOwner,
	std::vector<Json>& _snapshots)
{
	size_t const prefixSize = _snapshots.size();
	std::vector<Json> trueSnapshots = _snapshots;
	std::vector<Json> falseSnapshots = _snapshots;
	auto trueTarget = resolveCalleeReturnPath(_trueExpr, _paramBinding, _snapshotOwner, trueSnapshots);
	auto falseTarget = resolveCalleeReturnPath(_falseExpr, _paramBinding, _snapshotOwner, falseSnapshots);
	if (!trueTarget || !falseTarget || !storageRefTargetsEqual(*trueTarget, *falseTarget)
		|| trueSnapshots != falseSnapshots)
		return std::nullopt;
	appendNewStorageRefSnapshots(_snapshots, std::move(trueSnapshots), prefixSize);
	return trueTarget;
}
std::optional<StorageRefTarget> resolveCalleeReturnPath(
	Expression const& _expr,
	std::map<VariableDeclaration const*, Expression const*> const& _paramBinding,
	ASTNode const& _snapshotOwner,
	std::vector<Json>& _snapshots)
{
	if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
		return resolveSameCalleeStorageRefBranches(
			conditional->trueExpression(), conditional->falseExpression(), _paramBinding, _snapshotOwner, _snapshots);
	if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr))
	{
		if (!indexAccess->indexExpression())
			return std::nullopt;
		Type const* baseType = indexAccess->baseExpression().annotation().type;
		if (!baseType)
			return std::nullopt;
		bool isArrayLike
			= baseType->category() == Type::Category::Array || baseType->category() == Type::Category::FixedBytes;
		bool isMapping = baseType->category() == Type::Category::Mapping;
		if (!isArrayLike && !isMapping)
			return std::nullopt;
		// The key lives in CALLEE scope: only a literal (context-free) or a
		// bare parameter identifier (whose caller-side argument expression is
		// substituted) can be exported soundly from the caller's body.
		Expression const* calleeKey = indexAccess->indexExpression();
		Expression const* keyExpr = nullptr;
		if (dynamic_cast<Literal const*>(calleeKey))
			keyExpr = calleeKey;
		else if (auto const* keyIdent = dynamic_cast<Identifier const*>(calleeKey))
		{
			auto const* keyDecl
				= dynamic_cast<VariableDeclaration const*>(keyIdent->annotation().referencedDeclaration);
			auto it = keyDecl ? _paramBinding.find(keyDecl) : _paramBinding.end();
			if (it == _paramBinding.end())
				return std::nullopt;
			keyExpr = it->second;
		}
		else
			return std::nullopt;
		auto base = resolveCalleeReturnPath(indexAccess->baseExpression(), _paramBinding, _snapshotOwner, _snapshots);
		if (!base)
			return std::nullopt;
		std::string tempName = storageRefKeyTempName(_snapshotOwner, _snapshots.size());
		Json letStmt = Json::object();
		letStmt["kind"] = "let";
		letStmt["sourceDeclarationId"] = Json();
		letStmt["name"] = tempName;
		Type const* keyType = calleeKey->annotation().type;
		auto simpleKeyType = keyType ? exportSimpleType(*keyType) : std::nullopt;
		letStmt["type"] = simpleKeyType.has_value() ? *simpleKeyType : Json("u256");
		letStmt["value"] = exportStorageRefSnapshotValue(*keyExpr, _snapshots);
		_snapshots.emplace_back(letStmt);
		base->steps.push_back(
			{isMapping ? StorageRefStep::Kind::MappingKey : StorageRefStep::Kind::ArrayIndex, "", tempName});
		return base;
	}
	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
	{
		auto base = resolveCalleeReturnPath(memberAccess->expression(), _paramBinding, _snapshotOwner, _snapshots);
		if (!base)
			return std::nullopt;
		base->steps.push_back({StorageRefStep::Kind::Field, memberAccess->memberName(), ""});
		return base;
	}
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
	{
		auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (!decl)
			return std::nullopt;
		if (decl->isStateVariable())
			return StorageRefTarget{decl->name(), {}, StorageRefTarget::RootKind::StateField, decl->annotation().type};
		auto it = _paramBinding.find(decl);
		if (it != _paramBinding.end())
			return resolveStorageRefPathRec(*it->second, _snapshotOwner, _snapshots);
		return std::nullopt;
	}
	return std::nullopt;
}

std::optional<StorageRefTarget> resolveCalleeSuccessfulReturnPath(
	FunctionDefinition const& _callee,
	std::map<VariableDeclaration const*, Expression const*> const& _paramBinding,
	ASTNode const& _snapshotOwner,
	std::vector<Json>& _snapshots)
{
	VariableDeclaration const* returnSlot = _callee.returnParameters().front().get();
	std::optional<StorageRefTarget> resolvedTarget;
	std::optional<std::vector<Json>> resolvedSnapshots;
	bool invalid = false;

	auto recordPath = [&](Expression const& expression)
	{
		if (invalid)
			return;
		std::vector<Json> candidateSnapshots = _snapshots;
		auto candidate = resolveCalleeReturnPath(expression, _paramBinding, _snapshotOwner, candidateSnapshots);
		if (!candidate)
		{
			invalid = true;
			return;
		}
		if (!resolvedTarget)
		{
			resolvedTarget = std::move(candidate);
			resolvedSnapshots = std::move(candidateSnapshots);
			return;
		}
		if (!storageRefTargetsEqual(*resolvedTarget, *candidate) || *resolvedSnapshots != candidateSnapshots)
			invalid = true;
	};

	std::function<void(Statement const&)> scanStatement;
	scanStatement = [&](Statement const& statement)
	{
		if (invalid)
			return;
		if (auto const* block = dynamic_cast<Block const*>(&statement))
		{
			for (auto const& nested: block->statements())
				scanStatement(*nested);
			return;
		}
		if (auto const* ifStatement = dynamic_cast<IfStatement const*>(&statement))
		{
			scanStatement(ifStatement->trueStatement());
			if (ifStatement->falseStatement())
				scanStatement(*ifStatement->falseStatement());
			return;
		}
		if (auto const* returnStatement = dynamic_cast<Return const*>(&statement))
		{
			if (!returnStatement->expression())
				return;
			if (auto const* identifier = dynamic_cast<Identifier const*>(returnStatement->expression()))
				if (identifier->annotation().referencedDeclaration == returnSlot)
					return;
			recordPath(*returnStatement->expression());
			return;
		}
		if (auto const* expressionStatement = dynamic_cast<ExpressionStatement const*>(&statement))
		{
			auto const* assignment = dynamic_cast<Assignment const*>(&expressionStatement->expression());
			if (!assignment)
				return;
			auto const* lhs = dynamic_cast<Identifier const*>(&assignment->leftHandSide());
			if (!lhs || lhs->annotation().referencedDeclaration != returnSlot)
				return;
			if (assignment->assignmentOperator() != Token::Assign)
			{
				invalid = true;
				return;
			}
			recordPath(assignment->rightHandSide());
		}
	};

	scanStatement(_callee.body());
	if (invalid || !resolvedTarget || !resolvedSnapshots)
		return std::nullopt;
	_snapshots = std::move(*resolvedSnapshots);
	return resolvedTarget;
}

/// [E1 — call-return storage-ref path propagation] Resolve a call-rooted
/// storage-reference initializer to the storage path the callee returns.
/// Sound only under ALL of the following, each checked fail-closed:
///   - the callee is a statically exact reference: an ordinary, implemented,
///     NON-virtual FunctionDefinition (a virtual callee could dispatch to an
///     override returning a different path);
///   - the callee is view/pure AND its body is EXACTLY ONE statement, a
///     `return <expr>;` — the call itself is never emitted (every use
///     re-projects the substituted path), so the callee must have no other
///     effect to drop;
///   - every caller argument (including a using-for bound receiver) is
///     side-effect-free: substituted arguments are re-evaluated at snapshot
///     position and unused arguments are dropped entirely;
///   - the returned expression resolves under `resolveCalleeReturnPath`
///     above (single static path; multiple/conditional returns never get
///     here because the body must be that one return statement).
/// The composed target then flows through the SAME bind-site guards as a
/// direct bind (StorageRefShrinkScanner's E0(b) shrink refusal), so the
/// early-bind divergence class stays guarded. Like the landed direct-bind
/// model, a bind whose snapshot keys are never used performs no bind-time
/// bounds check (reads re-project at use sites) — E1 adds no new divergence
/// class beyond that landed discipline.
///
/// Assembly-derived storage-slot helpers resolve first through their exact
/// Yul assignment provenance.  Ordinary call-returned paths then retain the
/// conservative effect-preserving substitution below.
std::optional<StorageRefTarget>
resolveStorageRefCallRoot(FunctionCall const& _call, ASTNode const& _snapshotOwner, std::vector<Json>& _snapshots)
{
	if (_call.annotation().kind.set() && *_call.annotation().kind != FunctionCallKind::FunctionCall)
		return std::nullopt;
	FunctionDefinition const* callee = nullptr;
	Expression const* receiver = nullptr;
	if (auto const* ident = dynamic_cast<Identifier const*>(&_call.expression()))
		callee = dynamic_cast<FunctionDefinition const*>(ident->annotation().referencedDeclaration);
	else if (auto const* member = dynamic_cast<MemberAccess const*>(&_call.expression()))
	{
		callee = dynamic_cast<FunctionDefinition const*>(member->annotation().referencedDeclaration);
		if (auto const* fnType = dynamic_cast<FunctionType const*>(_call.expression().annotation().type))
			if (fnType->hasBoundFirstArgument())
				receiver = &member->expression();
	}
	if (!callee || !callee->isImplemented() || !callee->isOrdinary())
		return std::nullopt;
	if (callee->virtualSemantics())
		return std::nullopt;
	if (callee->stateMutability() != StateMutability::View && callee->stateMutability() != StateMutability::Pure)
		return std::nullopt;
	if (callee->returnParameters().size() != 1)
		return std::nullopt;
	{
		Type const* retType = callee->returnParameters().front()->annotation().type;
		bool storageRefReturn = false;
		if (auto const* refType = dynamic_cast<ReferenceType const*>(retType))
			storageRefReturn = refType->location() == DataLocation::Storage;
		else if (retType && retType->category() == Type::Category::Mapping)
			storageRefReturn = true;
		if (!storageRefReturn)
			return std::nullopt;
	}
	std::vector<Expression const*> callerArgs;
	if (receiver)
		callerArgs.push_back(receiver);
	for (auto const& arg: _call.arguments())
		callerArgs.push_back(arg.get());
	if (callerArgs.size() != callee->parameters().size())
		return std::nullopt;
	std::map<VariableDeclaration const*, Expression const*> paramBinding;
	for (size_t i = 0; i < callerArgs.size(); ++i)
	{
		if (!callerArgs[i])
			return std::nullopt;
		paramBinding[callee->parameters()[i].get()] = callerArgs[i];
	}
	if (auto rawSlotTarget
		= resolveAssemblyRawSlotReturnPath(*callee, paramBinding, _snapshotOwner, _snapshots))
		return rawSlotTarget;
	for (auto const* arg: callerArgs)
		if (!isSideEffectFreeExpr(*arg))
			return std::nullopt;
	auto target = resolveCalleeSuccessfulReturnPath(*callee, paramBinding, _snapshotOwner, _snapshots);
	if (!target)
		return std::nullopt;

	// Static substitution carries path identity, but it must not erase the
	// call's dynamic success/failure. Evaluate the real call once as a
	// discarded expression before using the frozen rooted path; Base/Analysis
	// retain its Result bind and state threading like any internal-call stmt.
	Json effectStatement = Json::object();
	effectStatement["kind"] = "expr";
	{
		StorageRefCallResolutionGuard guard;
		effectStatement["value"] = exportExpr(_call);
	}
	_snapshots.emplace_back(std::move(effectStatement));
	return target;
}

/// Recursive core of `resolveStorageRefInitializer`: resolve `_expr` to a
/// rooted storage path, appending key-snapshot `let`s to `_snapshots` in
/// root-outward (evaluation) order. Shared with the [E1] call-root
/// resolution above, whose parameter substitution recurses back into this
/// function for caller-side argument paths (snapshot numbering stays
/// collision-free because both append to the same `_snapshots` vector).
std::optional<StorageRefTarget>
resolveStorageRefPathRec(Expression const& _expr, ASTNode const& _snapshotOwner, std::vector<Json>& _snapshots)
{
	if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr))
	{
		if (!indexAccess->indexExpression())
			return std::nullopt;
		Type const* baseType = indexAccess->baseExpression().annotation().type;
		if (!baseType)
			return std::nullopt;
		bool isArrayLike
			= baseType->category() == Type::Category::Array || baseType->category() == Type::Category::FixedBytes;
		bool isMapping = baseType->category() == Type::Category::Mapping;
		if (!isArrayLike && !isMapping)
			return std::nullopt;
		auto base = resolveStorageRefPathRec(indexAccess->baseExpression(), _snapshotOwner, _snapshots);
		if (!base)
			return std::nullopt;
		std::string tempName = storageRefKeyTempName(_snapshotOwner, _snapshots.size());
		Json letStmt = Json::object();
		letStmt["kind"] = "let";
		letStmt["sourceDeclarationId"] = Json();
		letStmt["name"] = tempName;
		Type const* keyType = indexAccess->indexExpression()->annotation().type;
		auto simpleKeyType = keyType ? exportSimpleType(*keyType) : std::nullopt;
		letStmt["type"] = simpleKeyType.has_value() ? *simpleKeyType : Json("u256");
		letStmt["value"] = exportStorageRefSnapshotValue(*indexAccess->indexExpression(), _snapshots);
		_snapshots.emplace_back(letStmt);
		base->steps.push_back(
			{isMapping ? StorageRefStep::Kind::MappingKey : StorageRefStep::Kind::ArrayIndex, "", tempName});
		return base;
	}
	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
	{
		// Namespaced-alias / getter-call bases: root at the flattened
		// prefix+member field instead of recursing further (mirrors the
		// read-site handling at the namespaced-storage member access).
		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			auto nsIt = namespacedStorageAliases.find(baseIdent->name());
			if (nsIt != namespacedStorageAliases.end())
				return StorageRefTarget{
					nsIt->second + memberAccess->memberName(),
					{},
					StorageRefTarget::RootKind::StateField,
					memberAccess->annotation().type};
		}
		if (auto const* call = dynamic_cast<FunctionCall const*>(&memberAccess->expression()))
			if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
				if (auto const* prefix = namespacedGetterPrefix(*callee))
					return StorageRefTarget{
						*prefix + memberAccess->memberName(),
						{},
						StorageRefTarget::RootKind::StateField,
						memberAccess->annotation().type};
		auto base = resolveStorageRefPathRec(memberAccess->expression(), _snapshotOwner, _snapshots);
		if (!base)
			return std::nullopt;
		base->steps.push_back({StorageRefStep::Kind::Field, memberAccess->memberName(), ""});
		return base;
	}
	if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
	{
		if (!isSideEffectFreeExpr(conditional->condition()))
			return std::nullopt;
		size_t const prefixSize = _snapshots.size();
		std::vector<Json> trueSnapshots = _snapshots;
		std::vector<Json> falseSnapshots = _snapshots;
		auto trueTarget = resolveStorageRefPathRec(conditional->trueExpression(), _snapshotOwner, trueSnapshots);
		auto falseTarget = resolveStorageRefPathRec(conditional->falseExpression(), _snapshotOwner, falseSnapshots);
		if (!trueTarget || !falseTarget || !storageRefTargetsEqual(*trueTarget, *falseTarget)
			|| trueSnapshots != falseSnapshots)
			return std::nullopt;

		Json conditionLet = Json::object();
		conditionLet["kind"] = "let";
		conditionLet["sourceDeclarationId"] = Json();
		conditionLet["name"] = "__solcore_sref_condition_" + std::to_string(stableSyntheticNodeId(_snapshotOwner));
		conditionLet["type"] = "bool";
		conditionLet["value"] = exportExpr(conditional->condition());
		_snapshots.emplace_back(std::move(conditionLet));
		appendNewStorageRefSnapshots(_snapshots, std::move(trueSnapshots), prefixSize);
		return trueTarget;
	}

	// [E1] Call-rooted paths: resolve single-return view/pure callees to the
	// path they return (see resolveStorageRefCallRoot above). Unresolvable
	// calls keep the copy-`let` fallback (§3.6 fallback policy), whose
	// write-through guard (E0(a')) still refuses any write through the
	// resulting untracked local.
	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
	{
		if (auto const* member = dynamic_cast<MemberAccess const*>(&call->expression()))
		{
			auto const* functionType = dynamic_cast<FunctionType const*>(member->annotation().type);
			auto const* arrayType = dynamic_cast<ArrayType const*>(member->expression().annotation().type);
			if (member->memberName() == "push" && functionType && functionType->kind() == FunctionType::Kind::ArrayPush
				&& arrayType && arrayType->isDynamicallySized() && call->arguments().empty())
			{
				auto base = resolveStorageRefPathRec(member->expression(), _snapshotOwner, _snapshots);
				if (!base)
					return std::nullopt;

				std::string indexName = "__solcore_push_last_" + std::to_string(stableSyntheticNodeId(*call));
				Json indexLet = Json::object();
				indexLet["kind"] = "let";
				indexLet["sourceDeclarationId"] = Json();
				indexLet["name"] = indexName;
				indexLet["type"] = "u256";
				Json length = Json::object();
				length["kind"] = "array_length";
				length["base"] = aliasReadJson(*base);
				indexLet["value"] = std::move(length);
				_snapshots.emplace_back(std::move(indexLet));

				Json pushStatement = Json::object();
				pushStatement["kind"] = "expr";
				Json push = Json::object();
				push["kind"] = "internal_call";
				push["function"] = "array_push_expr";
				push["args"] = Json::array();
				push["args"].emplace_back(aliasReadJson(*base));
				Json defaultValue = Json::object();
				defaultValue["kind"] = "typed_default";
				defaultValue["type"] = exportResolvedType(arrayType->baseType());
				push["args"].emplace_back(std::move(defaultValue));
				pushStatement["value"] = std::move(push);
				_snapshots.emplace_back(std::move(pushStatement));

				base->steps.push_back({StorageRefStep::Kind::ArrayIndex, "", indexName});
				return base;
			}
		}
		return resolveStorageRefCallRoot(*call, _snapshotOwner, _snapshots);
	}
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
	{
		auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (!decl)
			return std::nullopt;
		if (decl->isStateVariable())
			return StorageRefTarget{decl->name(), {}, StorageRefTarget::RootKind::StateField, decl->annotation().type};
		if (isStorageRefParameter(decl))
		{
			StorageRefTarget target{decl->name(), {}};
			target.rootKind = StorageRefTarget::RootKind::LocalParameter;
			target.rootType = decl->annotation().type;
			return target;
		}
		// Composition: aliasing another TRACKED, RESOLVED alias.
		auto it = storageRefAliasTargets.find(decl);
		if (it != storageRefAliasTargets.end())
			return it->second;
		// An untracked/unresolvable storage local is still a value copy:
		// fail closed rather than promoting it to a first-class reference.
		return std::nullopt;
	}
	// Ternary and any other expression shape: not resolvable (§0's explicit
	// Phase-1 boundary).
	return std::nullopt;
}

/// §3.2 bind-site resolution: walk a storage-located local's initializer
/// expression and, if it takes one of the recognized shapes, produce the
/// alias's StorageRefTarget plus the key-snapshot `let` statements that must
/// be emitted (in order) at the bind site (design §3.2/§7.1: the key is
/// evaluated ONCE, at bind time, matching Solidity's own "slot computed at
/// declaration" semantics). Returns nullopt for any shape not covered
/// (ternary/unresolvable-call/etc. — §3.6's fallback policy: the caller then
/// falls through to today's status-quo copy-`let` lowering, unchanged).
std::optional<std::pair<StorageRefTarget, std::vector<Json>>>
resolveStorageRefInitializer(Expression const& _init, VariableDeclaration const& _bindDecl)
{
	std::vector<Json> snapshots;
	auto target = resolveStorageRefPathRec(_init, _bindDecl, snapshots);
	if (!target)
		return std::nullopt;
	return std::make_pair(std::move(*target), std::move(snapshots));
}

/// Resolve one storage-path USE (write base, receiver, or push/pop base) and
/// commit its dynamic key/index captures to the enclosing statement exactly
/// once. Bind-site resolution uses the same recursive producer above but owns
/// its snapshot block explicitly; this helper is only for use sites.
///
/// Stable-only users retain the resolver's conservative side-effect-free
/// rule. A statement-isolated delete may snapshot the key of one exact state
/// mapping access. Builtin storage-array push/pop use OrderedPath: both solc
/// pipelines evaluate the bound receiver (base, then keys/indices) before the
/// value argument, so every dynamic path component is captured root-outward
/// before that argument is pinned. The functional update then reads through
/// the frozen location and Base.ml writes it back through the same path.
/// Call-returned paths still enforce their own duplication-safe argument rule
/// inside [resolveStorageRefCallRoot].
std::optional<Json>
exportResolvedStorageRefUse(Expression const& _expr, StorageRefKeySnapshotMode _snapshotMode)
{
	bool const snapshotOrderedPath = _snapshotMode == StorageRefKeySnapshotMode::OrderedPath;
	bool snapshotDirectMappingKey = false;
	if (_snapshotMode == StorageRefKeySnapshotMode::DirectMappingKey)
	{
		auto const* index = dynamic_cast<IndexAccess const*>(&_expr);
		auto const* root = index ? dynamic_cast<Identifier const*>(&index->baseExpression()) : nullptr;
		auto const* decl = root
							   ? dynamic_cast<VariableDeclaration const*>(root->annotation().referencedDeclaration)
							   : nullptr;
		snapshotDirectMappingKey
			= index && index->indexExpression() && decl && decl->isStateVariable()
			  && dynamic_cast<MappingType const*>(index->baseExpression().annotation().type);
	}

	std::function<bool(Expression const&)> keysAreAdmissible = [&](Expression const& expr) -> bool
	{
		if (auto const* index = dynamic_cast<IndexAccess const*>(&expr))
			return index->indexExpression()
				   && (snapshotOrderedPath || (snapshotDirectMappingKey && &expr == &_expr)
					   || isSideEffectFreeExpr(*index->indexExpression()))
				   && keysAreAdmissible(index->baseExpression());
		if (auto const* member = dynamic_cast<MemberAccess const*>(&expr))
			return keysAreAdmissible(member->expression());
		if (auto const* tuple = dynamic_cast<TupleExpression const*>(&expr))
			return !tuple->isInlineArray() && tuple->components().size() == 1 && tuple->components().front()
				   && keysAreAdmissible(*tuple->components().front());
		if (auto const* conditional = dynamic_cast<Conditional const*>(&expr))
			return isSideEffectFreeExpr(conditional->condition())
				   && keysAreAdmissible(conditional->trueExpression())
				   && keysAreAdmissible(conditional->falseExpression());
		return dynamic_cast<Identifier const*>(&expr) || dynamic_cast<FunctionCall const*>(&expr);
	};
	if (!keysAreAdmissible(_expr))
		return std::nullopt;

	std::vector<Json> snapshots;
	auto target = resolveStorageRefPathRec(_expr, _expr, snapshots);
	if (!target)
		return std::nullopt;
	if (target->snapshotAsTypedCall)
	{
		// The assembly helper reinterprets its captured root slot as a different
		// storage referent. Alias-read substitution would return the caller
		// root's type; retain the ordinary typed internal call instead.
		return std::nullopt;
	}
	if (!snapshots.empty())
	{
		if (!activeHoistScope || !activeHoistScope->allowed)
			return std::nullopt;
		int64_t const ownerId = static_cast<int64_t>(_expr.id());
		if (activeHoistScope->storageRefSnapshotOwners.insert(ownerId).second)
			for (Json& snapshot: snapshots)
				activeHoistScope->statements.emplace_back(std::move(snapshot));
	}
	return aliasReadJson(*target);
}

// --- Residual first-class storage-reference values (U14) ---
//
// Static aliases above remain the canonical zero-runtime-cost path. The
// helpers below are used only when path substitution cannot carry the source
// semantics: conditional selection, rebinding, a conditional library return,
// or a public-library ABI slot word.

bool isStorageReferenceType(Type const* _type)
{
	if (!_type)
		return false;
	if (_type->category() == Type::Category::Mapping)
		return true;
	auto const* reference = dynamic_cast<ReferenceType const*>(_type);
	return reference && reference->location() == DataLocation::Storage;
}

bool hasSingleStorageReferenceReturn(FunctionDefinition const& _function)
{
	return _function.returnParameters().size() == 1
		   && isStorageReferenceType(_function.returnParameters().front()->annotation().type);
}

void collectStorageRefPathSignatures(Expression const& _expr, std::set<std::string>& _out)
{
	if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
	{
		collectStorageRefPathSignatures(conditional->trueExpression(), _out);
		collectStorageRefPathSignatures(conditional->falseExpression(), _out);
		return;
	}
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
	{
		if (auto const* declaration = identifier->annotation().referencedDeclaration)
			_out.insert("d:" + std::to_string(static_cast<int64_t>(declaration->id())));
		return;
	}
	if (auto const* member = dynamic_cast<MemberAccess const*>(&_expr))
	{
		std::set<std::string> bases;
		collectStorageRefPathSignatures(member->expression(), bases);
		for (std::string const& base: bases)
			_out.insert(base + ".f:" + member->memberName());
		return;
	}
	if (auto const* index = dynamic_cast<IndexAccess const*>(&_expr))
	{
		if (!index->indexExpression())
			return;
		std::set<std::string> bases;
		std::set<std::string> keys;
		collectStorageRefPathSignatures(index->baseExpression(), bases);
		collectStorageRefPathSignatures(*index->indexExpression(), keys);
		for (std::string const& base: bases)
			for (std::string const& key: keys)
				_out.insert(base + "[k:" + key + "]");
		return;
	}
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expr))
		if (!tuple->isInlineArray() && tuple->components().size() == 1 && tuple->components().front())
			collectStorageRefPathSignatures(*tuple->components().front(), _out);
}

bool storageRefExpressionRootedAtLocal(Expression const& _expr)
{
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
		if (auto const* declaration
			= dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration))
			return !declaration->isStateVariable() && !isStorageRefParameter(declaration)
				   && !declaration->isReturnParameter()
				   && declaration->referenceLocation() == VariableDeclaration::Location::Storage;
	if (auto const* member = dynamic_cast<MemberAccess const*>(&_expr))
		return storageRefExpressionRootedAtLocal(member->expression());
	if (auto const* index = dynamic_cast<IndexAccess const*>(&_expr))
		return storageRefExpressionRootedAtLocal(index->baseExpression());
	if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
		return storageRefExpressionRootedAtLocal(conditional->trueExpression())
			   || storageRefExpressionRootedAtLocal(conditional->falseExpression());
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expr))
		return !tuple->isInlineArray() && tuple->components().size() == 1 && tuple->components().front()
			   && storageRefExpressionRootedAtLocal(*tuple->components().front());
	return false;
}

struct ResidualStorageRefReturnScanner: ASTConstVisitor
{
	VariableDeclaration const* returnSlot;
	std::set<std::string> paths;
	bool usesLocalRoot = false;

	explicit ResidualStorageRefReturnScanner(VariableDeclaration const* _returnSlot): returnSlot(_returnSlot) {}

	bool visit(Return const& _return) override
	{
		if (_return.expression())
		{
			collectStorageRefPathSignatures(*_return.expression(), paths);
			usesLocalRoot = usesLocalRoot || storageRefExpressionRootedAtLocal(*_return.expression());
		}
		return true;
	}

	bool visit(Assignment const& _assignment) override
	{
		if (_assignment.assignmentOperator() == Token::Assign && returnSlot)
			if (auto const* lhs = dynamic_cast<Identifier const*>(&_assignment.leftHandSide()))
				if (lhs->annotation().referencedDeclaration == returnSlot)
				{
					collectStorageRefPathSignatures(_assignment.rightHandSide(), paths);
					usesLocalRoot = usesLocalRoot || storageRefExpressionRootedAtLocal(_assignment.rightHandSide());
				}
		return true;
	}
};

bool isResidualStorageRefFunction(FunctionDefinition const& _function)
{
	if (!hasSingleStorageReferenceReturn(_function))
		return false;
	auto const* scope = dynamic_cast<ContractDefinition const*>(_function.scope());
	if (!scope || !scope->isLibrary())
		return false;
	ResidualStorageRefReturnScanner scanner(_function.returnParameters().front().get());
	_function.body().accept(scanner);
	return scanner.paths.size() > 1 || scanner.usesLocalRoot;
}

struct ResidualStorageRefLocalScanner: ASTConstVisitor
{
	bool needed = false;

	bool visit(VariableDeclarationStatement const& _statement) override
	{
		if (_statement.initialValue())
			for (auto const& declaration: _statement.declarations())
				if (declaration && declaration->referenceLocation() == VariableDeclaration::Location::Storage
					&& dynamic_cast<Conditional const*>(_statement.initialValue()))
					needed = true;
		return !needed;
	}

	bool visit(Assignment const& _assignment) override
	{
		if (auto const* lhs = dynamic_cast<Identifier const*>(&_assignment.leftHandSide()))
			if (auto const* declaration
				= dynamic_cast<VariableDeclaration const*>(lhs->annotation().referencedDeclaration))
				if (!declaration->isStateVariable()
					&& declaration->referenceLocation() == VariableDeclaration::Location::Storage)
					needed = true;
		return !needed;
	}
};

bool functionNeedsResidualStorageRefLocals(FunctionDefinition const& _function)
{
	ResidualStorageRefLocalScanner scanner;
	_function.body().accept(scanner);
	return scanner.needed;
}

bool isPublicLibraryStructuralStorageFunction(FunctionDefinition const& _function)
{
	auto const* scope = dynamic_cast<ContractDefinition const*>(_function.scope());
	bool const hasStorageParameter = std::any_of(
		_function.parameters().begin(),
		_function.parameters().end(),
		[](auto const& parameter) { return isStorageRefParameter(parameter.get()); });
	return scope && scope->isLibrary() && hasStorageParameter
		   && (_function.visibility() == Visibility::Public || _function.visibility() == Visibility::External)
		   && !hasSingleStorageReferenceReturn(_function);
}

std::string storageRefInternalEntryName(FunctionDefinition const& _function)
{
	return exportedFunctionName(_function) + "__storage_ref";
}

VariableDeclaration const* storageRefRootParameter(Expression const& _expression)
{
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expression))
	{
		auto const* declaration
			= dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		return isStorageRefParameter(declaration) ? declaration : nullptr;
	}
	if (auto const* member = dynamic_cast<MemberAccess const*>(&_expression))
		return storageRefRootParameter(member->expression());
	if (auto const* index = dynamic_cast<IndexAccess const*>(&_expression))
		return storageRefRootParameter(index->baseExpression());
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expression))
		if (!tuple->isInlineArray() && tuple->components().size() == 1 && tuple->components().front())
			return storageRefRootParameter(*tuple->components().front());
	return nullptr;
}

struct ResidualStorageRefArgumentScanner: ASTConstVisitor
{
	std::set<VariableDeclaration const*> parameters;

	bool visit(FunctionCall const& _call) override
	{
		FunctionDefinition const* target = nullptr;
		if (auto const* identifier = dynamic_cast<Identifier const*>(&_call.expression()))
			target = dynamic_cast<FunctionDefinition const*>(identifier->annotation().referencedDeclaration);
		else if (auto const* member = dynamic_cast<MemberAccess const*>(&_call.expression()))
			target = dynamic_cast<FunctionDefinition const*>(member->annotation().referencedDeclaration);
		if (target && hasSingleStorageReferenceReturn(*target))
			for (auto const& argument: _call.arguments())
				if (VariableDeclaration const* parameter = storageRefRootParameter(*argument))
					parameters.insert(parameter);
		return true;
	}
};

bool storageRefParameterFeedsResidualReference(
	FunctionDefinition const& _function, VariableDeclaration const* _parameter)
{
	if (!isStorageRefParameter(_parameter))
		return false;
	ResidualStorageRefArgumentScanner scanner;
	_function.body().accept(scanner);
	return scanner.parameters.count(_parameter) != 0;
}

bool storageRefParameterIsWrittenThrough(
	FunctionDefinition const& _function, VariableDeclaration const* _parameter)
{
	if (!isStorageRefParameter(_parameter))
		return false;
	StorageRefWriteThroughScanner writeThrough;
	_function.body().accept(writeThrough);
	return writeThrough.writtenThrough.count(_parameter) != 0;
}

bool functionNeedsAllStructuralStorageRefs(FunctionDefinition const& _function)
{
	return isResidualStorageRefFunction(_function) || isPublicLibraryStructuralStorageFunction(_function)
		   || functionNeedsResidualStorageRefLocals(_function);
}


bool isStructuralStorageRefParameter(FunctionDefinition const& _function, VariableDeclaration const* _parameter)
{
	return isStorageRefParameter(_parameter)
		   && (functionNeedsAllStructuralStorageRefs(_function)
			   || storageRefParameterIsWrittenThrough(_function, _parameter)
			   || storageRefParameterFeedsResidualReference(_function, _parameter));
}

FunctionDefinition const* calledFunctionDefinition(FunctionCall const& _call)
{
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_call.expression()))
		return dynamic_cast<FunctionDefinition const*>(identifier->annotation().referencedDeclaration);
	if (auto const* member = dynamic_cast<MemberAccess const*>(&_call.expression()))
		return dynamic_cast<FunctionDefinition const*>(member->annotation().referencedDeclaration);
	return nullptr;
}

Json storageRefWireType(Type const* _referentType)
{
	Json result = Json::object();
	result["kind"] = "storage_ref";
	result["referent"] = exportResolvedType(_referentType, true);
	return result;
}

Json storageRefGetJson(Json _reference, Type const* _referentType)
{
	Json result = Json::object();
	result["kind"] = "storage_ref_get";
	result["referentType"] = exportResolvedType(_referentType, true);
	result["reference"] = std::move(_reference);
	return result;
}

Type const* storageRefFieldType(Type const* _containerType, std::string const& _field)
{
	auto const* structType = dynamic_cast<StructType const*>(_containerType);
	if (!structType)
		throw UnsupportedSolCore("Residual storage-reference field projection has a non-struct container.");
	for (auto const& member: structType->structDefinition().members())
		if (member->name() == _field)
			return member->annotation().type;
	throw UnsupportedSolCore(
		"Residual storage-reference field projection has no compiler-resolved member '" + _field + "'.");
}

Json storageRefFromTarget(StorageRefTarget const& _target)
{
	if (!_target.rootType)
		throw UnsupportedSolCore("Residual storage-reference target has no compiler-resolved root type.");
	Json current = Json::object();
	if (_target.rootKind == StorageRefTarget::RootKind::LocalParameter)
		current = localExpr(_target.root);
	else if (_target.rootKind == StorageRefTarget::RootKind::RawSlot)
	{
		if (_target.rawSlot.is_null())
			throw UnsupportedSolCore("Raw-slot storage-reference target has no slot-word provenance.");
		current["kind"] = "storage_ref_raw_slot";
		current["referentType"] = exportResolvedType(_target.rootType, true);
		current["slot"] = _target.rawSlot;
	}
	else
	{
		current["kind"] = "storage_ref_root";
		current["referentType"] = exportResolvedType(_target.rootType, true);
		current["field"] = _target.root;
	}
	Type const* currentType = _target.rootType;
	for (StorageRefStep const& step: _target.steps)
	{
		Json next = Json::object();
		Type const* referentType = nullptr;
		if (step.kind == StorageRefStep::Kind::Field)
		{
			referentType = storageRefFieldType(currentType, step.field);
			next["kind"] = "storage_ref_field";
			next["field"] = step.field;
		}
		else if (step.kind == StorageRefStep::Kind::MappingKey)
		{
			auto const* mappingType = dynamic_cast<MappingType const*>(currentType);
			if (!mappingType)
				throw UnsupportedSolCore("Residual storage-reference mapping projection has a non-mapping container.");
			referentType = mappingType->valueType();
			next["kind"] = "storage_ref_mapping";
			next["key"] = localExpr(step.keyTemp);
		}
		else
		{
			auto const* arrayType = dynamic_cast<ArrayType const*>(currentType);
			if (!arrayType)
				throw UnsupportedSolCore("Residual storage-reference array projection has a non-array container.");
			referentType = arrayType->baseType();
			next["kind"] = "storage_ref_array";
			next["index"] = localExpr(step.keyTemp);
		}
		next["containerType"] = exportResolvedType(currentType, true);
		next["referentType"] = exportResolvedType(referentType, true);
		next["base"] = std::move(current);
		current = std::move(next);
		currentType = referentType;
	}
	return current;
}

/// Resolve an ordered storage lvalue to its first-class reference while
/// committing the resolver's path snapshots and mutations exactly once.
/// This is deliberately narrower than [exportResolvedStorageRefUse]: callers
/// need the reference identity itself, not a value read through that path.
std::optional<Json>
exportOrderedStorageRefValueUse(Expression const& _expr, bool* _snapshotAsTypedCall = nullptr)
{
	std::vector<Json> snapshots;
	auto target = resolveStorageRefPathRec(_expr, _expr, snapshots);
	if (!target)
		return std::nullopt;
	if (target->snapshotAsTypedCall)
	{
		if (_snapshotAsTypedCall)
			*_snapshotAsTypedCall = true;
		// The actual pure call owns evaluation and its correctly typed return;
		// path-resolution snapshots would duplicate that evaluation.
		return Json::object();
	}
	if (!snapshots.empty())
	{
		if (!activeHoistScope || !activeHoistScope->allowed)
			return std::nullopt;
		int64_t const ownerId = static_cast<int64_t>(_expr.id());
		if (activeHoistScope->storageRefSnapshotOwners.insert(ownerId).second)
			for (Json& snapshot: snapshots)
				activeHoistScope->statements.emplace_back(std::move(snapshot));
	}
	return storageRefFromTarget(*target);
}

static thread_local bool exportingStorageRefValueExpression = false;

struct StorageRefValueExpressionScope
{
	bool saved;
	StorageRefValueExpressionScope(): saved(exportingStorageRefValueExpression)
	{
		exportingStorageRefValueExpression = true;
	}
	~StorageRefValueExpressionScope() { exportingStorageRefValueExpression = saved; }
};

std::optional<Json> exportStorageRefValue(Expression const& _expr)
{
	// Assignment-LHS snapshot invariant: an assembly `.slot` helper whose
	// return referent differs from its captured root must remain the actual
	// typed pure call. Check before the active-hoist memo: that memo deliberately
	// aliases the call to its resolved caller root and would recreate the
	// ill-typed `StorageRef<root> : StorageRef<return>` Let.
	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
	{
		std::vector<Json> probeSnapshots;
		auto probe = resolveStorageRefCallRoot(*call, *call, probeSnapshots);
		if (probe && probe->snapshotAsTypedCall)
		{
			StorageRefValueExpressionScope scope;
			return exportExpr(*call);
		}
	}
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expr))
		if (!tuple->isInlineArray() && tuple->components().size() == 1 && tuple->components().front())
			return exportStorageRefValue(*tuple->components().front());

	if (activeHoistScope)
	{
		auto memo = activeHoistScope->memo.find(static_cast<int64_t>(_expr.id()));
		if (memo != activeHoistScope->memo.end())
			return localExpr(memo->second);
	}

	if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
	{
		auto trueReference = exportStorageRefValue(conditional->trueExpression());
		auto falseReference = exportStorageRefValue(conditional->falseExpression());
		if (!trueReference || !falseReference)
			return std::nullopt;
		Json result = Json::object();
		result["kind"] = "conditional";
		result["result_type"] = storageRefWireType(conditional->annotation().type);
		result["cond"] = exportExpr(conditional->condition());
		result["true_value"] = std::move(*trueReference);
		result["false_value"] = std::move(*falseReference);
		return result;
	}

	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
	{
		auto const* declaration
			= dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (!declaration)
			return std::nullopt;
		if (storageRefValueLocals.count(declaration))
			return localExpr(declaration->name().empty() ? identifier->name() : declaration->name());
		if (auto it = storageRefAliasTargets.find(declaration); it != storageRefAliasTargets.end())
			return storageRefFromTarget(it->second);
		if (declaration->isStateVariable())
		{
			Json result = Json::object();
			result["kind"] = "storage_ref_root";
			result["referentType"] = exportResolvedType(declaration->annotation().type, true);
			result["field"] = declaration->name();
			return result;
		}
		return std::nullopt;
	}

	if (auto const* member = dynamic_cast<MemberAccess const*>(&_expr))
	{
		if (auto const* baseIdentifier = dynamic_cast<Identifier const*>(&member->expression()))
		{
			auto namespaced = namespacedStorageAliases.find(baseIdentifier->name());
			if (namespaced != namespacedStorageAliases.end())
			{
				Json result = Json::object();
				result["kind"] = "storage_ref_root";
				result["referentType"] = exportResolvedType(member->annotation().type, true);
				result["field"] = namespaced->second + member->memberName();
				return result;
			}
		}
		auto base = exportStorageRefValue(member->expression());
		if (!base)
			return std::nullopt;
		Json result = Json::object();
		result["kind"] = "storage_ref_field";
		result["containerType"] = exportResolvedType(member->expression().annotation().type, true);
		result["referentType"] = exportResolvedType(member->annotation().type, true);
		result["base"] = std::move(*base);
		result["field"] = member->memberName();
		return result;
	}

	if (auto const* index = dynamic_cast<IndexAccess const*>(&_expr))
	{
		if (!index->indexExpression())
			return std::nullopt;
		Type const* containerType = index->baseExpression().annotation().type;
		if (!containerType)
			return std::nullopt;
		auto base = exportStorageRefValue(index->baseExpression());
		if (!base)
			return std::nullopt;
		Json result = Json::object();
		if (containerType->category() == Type::Category::Mapping)
		{
			result["kind"] = "storage_ref_mapping";
			result["key"] = exportExpr(*index->indexExpression());
		}
		else if (containerType->category() == Type::Category::Array)
		{
			result["kind"] = "storage_ref_array";
			result["index"] = exportExpr(*index->indexExpression());
		}
		else
			return std::nullopt;
		result["containerType"] = exportResolvedType(containerType, true);
		result["referentType"] = exportResolvedType(index->annotation().type, true);
		result["base"] = std::move(*base);
		return result;
	}

	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
	{
		bool snapshotAsTypedCall = false;
		if (auto reference = exportOrderedStorageRefValueUse(*call, &snapshotAsTypedCall))
		{
			if (snapshotAsTypedCall)
			{
				StorageRefValueExpressionScope scope;
				return exportExpr(*call);
			}
			return reference;
		}
		if (auto const* function = calledFunctionDefinition(*call))
			if (hasSingleStorageReferenceReturn(*function))
			{
				StorageRefValueExpressionScope scope;
				return exportExpr(*call);
			}
	}

	return std::nullopt;
}

bool hasResidualStorageRefRoot(Expression const& _expr)
{
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
	{
		auto const* declaration
			= dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		return declaration && storageRefValueLocals.count(declaration);
	}
	if (auto const* member = dynamic_cast<MemberAccess const*>(&_expr))
		return hasResidualStorageRefRoot(member->expression());
	if (auto const* index = dynamic_cast<IndexAccess const*>(&_expr))
		return hasResidualStorageRefRoot(index->baseExpression());
	if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
		return hasResidualStorageRefRoot(conditional->trueExpression())
			   || hasResidualStorageRefRoot(conditional->falseExpression());
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expr))
		return !tuple->isInlineArray() && tuple->components().size() == 1 && tuple->components().front()
			   && hasResidualStorageRefRoot(*tuple->components().front());
	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
		if (auto const* function = calledFunctionDefinition(*call))
			return hasSingleStorageReferenceReturn(*function);
	return false;
}

// --- End residual first-class storage-reference values ---

bool isCurrentUnitExternalReceiver(Expression const& _receiver)
{
	auto const* identifier = dynamic_cast<Identifier const*>(&_receiver);
	if (!identifier || identifier->name() != "this")
		return false;
	if (!dynamic_cast<MagicVariableDeclaration const*>(identifier->annotation().referencedDeclaration))
		return false;
	return dynamic_cast<ContractType const*>(identifier->annotation().type) != nullptr;
}

char const* externalCallMode(FunctionType const& _callType, bool _sameUnit)
{
	if (_sameUnit)
		return "same_unit";
	if (_callType.kind() == FunctionType::Kind::DelegateCall)
		return "delegatecall";
	if (_callType.kind() != FunctionType::Kind::External)
		throw UnsupportedSolCore(
			"High-level external call has compiler-resolved function kind '"
			+ _callType.richIdentifier() + "', not external or delegatecall.");
	return _callType.stateMutability() <= StateMutability::View ? "staticcall" : "call";
}

std::optional<Json>
exportExternalContractCall(FunctionCall const& _call, MemberAccess const& _memberAccess, bool _asStatement)
{
	Type const* baseType = _memberAccess.expression().annotation().type;
	if (!baseType || baseType->category() != Type::Category::Contract)
		return std::nullopt;

	auto const* callType = dynamic_cast<FunctionType const*>(_call.expression().annotation().type);
	if (!callType)
		throw UnsupportedSolCore(
			"High-level external call has no compiler-resolved call-expression function type.");
	auto const* options = dynamic_cast<FunctionCallOptions const*>(&_call.expression());
	CallEvaluation evaluation
		= exportCallEvaluation(_call, _memberAccess.expression(), options, callType);

	Json callExpr = Json::object();
	callExpr["target"] = std::move(evaluation.target);
	callExpr["method"] = _memberAccess.memberName();
	callExpr["callMode"] = externalCallMode(
		*callType, isCurrentUnitExternalReceiver(_memberAccess.expression()));
	callExpr["args"] = Json::array();
	for (Json& argument: evaluation.arguments)
		callExpr["args"].emplace_back(std::move(argument));

	bool const statefulCall = callType->stateMutability() > StateMutability::View;
	if (activeCompilerStack)
		if (auto knownTarget = exportKnownExternalTarget(*activeCompilerStack, _memberAccess))
			callExpr["knownTarget"] = std::move(*knownTarget);

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

std::string overloadSignatureSuffixOfParameters(std::vector<ASTPointer<VariableDeclaration>> const& _parameters)
{
	if (_parameters.empty())
		return "unit";

	std::string suffix;
	for (auto const& parameter: _parameters)
	{
		if (!suffix.empty())
			suffix += "_";
		if (!parameter->annotation().type)
			throw UnsupportedSolCore(
				"Overloaded event parameter has no compiler-resolved type for name disambiguation.");
		suffix += sanitizeExportNameComponent(parameter->annotation().type->toString());
	}
	return suffix;
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
		std::string typeName = parameter->annotation().type ? parameter->annotation().type->toString()
															: ("arg" + std::to_string(parameter->id()));
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
		if (candidate && candidate->isOrdinary() && candidate->isImplemented()
			&& (candidate->name().empty() ? "_unnamed" : candidate->name()) == rawName)
			matchingSignatures.insert(overloadSignatureSuffix(*candidate));
	}
	if (matchingSignatures.size() > 1)
		return rawName + "_" + overloadSignatureSuffix(_function);
	return rawName;
}

std::string contractScopedSuperAlias(FunctionDefinition const& _function)
{
	if (auto existing = contractScopedSuperAliases.find(&_function); existing != contractScopedSuperAliases.end())
		return existing->second;
	auto const* owner = dynamic_cast<ContractDefinition const*>(_function.scope());
	std::string ownerName = owner && !owner->name().empty() ? owner->name() : "Base";
	// This lookup occurs after overload assignment. Building the alias from
	// that exact identity keeps same-name/same-arity overloads distinct.
	return exportedFunctionName(_function) + "__super__" + sanitizeExportNameComponent(ownerName);
}

/// Scope-qualified exported name for an out-of-linearization library/free
/// function callee whose assigned name collides with another exported
/// function (spechunt-remaining-rejects-design §B1).
///
/// `_assignedName` is the name `assignExportedFunctionNames` ALREADY handed
/// this definition, not the raw source name: qualifying off the assigned name
/// preserves the typed-suffix disambiguation of two same-name overloads
/// *within* one library (`f_uint256` / `f_bytes32` stay distinct after
/// qualification; qualifying off the raw name would collapse them onto one
/// `f__lib__L`).
///
/// The `__lib__`/`__file__` double-underscore convention mirrors the existing
/// `contractScopedSuperAlias` (`__super__`) precedent and is a valid Lean
/// identifier fragment.
///
/// Returns nullopt when the callee is neither library-scoped nor file-scoped,
/// or when a file-scoped callee has no usable source name. There is no third
/// naming arm on purpose: the caller's post-qualification recheck then throws
/// exactly as it did before this qualification existed.
std::optional<std::string>
scopeQualifiedCalleeName(FunctionDefinition const& _function, std::string const& _assignedName)
{
	if (auto const* scopeContract = dynamic_cast<ContractDefinition const*>(_function.scope()))
	{
		// Out-of-linearization callees can come from distinct libraries with
		// the same source-level name.  Qualify by the full contract identity
		// so both their declarations and call sites remain unique downstream.
		if (!scopeContract->isLibrary())
			return std::nullopt;
		return _assignedName + "__lib__" + sanitizeExportNameComponent(exportedContractId(*scopeContract));
	}
	if (dynamic_cast<SourceUnit const*>(_function.scope()))
	{
		// `location().sourceName` is a plain shared_ptr — unlike
		// SourceUnitAnnotation::path (a SetOnce whose accessor throws when
		// unset), it is always safe to read. Use the BASENAME stem so an
		// exported name never embeds the producer's directory layout; two
		// same-stem files in different directories therefore remain a
		// possible second-order collision, which the caller's recheck refuses
		// loudly instead of papering over.
		std::shared_ptr<std::string const> const& sourceName = _function.location().sourceName;
		if (!sourceName || sourceName->empty())
			return std::nullopt;
		std::string stem = *sourceName;
		if (size_t slash = stem.find_last_of('/'); slash != std::string::npos)
			stem = stem.substr(slash + 1);
		if (stem.size() > 4 && stem.compare(stem.size() - 4, 4, ".sol") == 0)
			stem = stem.substr(0, stem.size() - 4);
		if (stem.empty())
			return std::nullopt;
		return _assignedName + "__file__" + sanitizeExportNameComponent(stem);
	}
	return std::nullopt;
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
	bool isSuper;
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
	FunctionCall const& _call, MemberAccess const& _memberAccess, ContractDefinition const* _mostDerived)
{
	auto const* typeType = dynamic_cast<TypeType const*>(_memberAccess.expression().annotation().type);
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
	auto const* funcDef = dynamic_cast<FunctionDefinition const*>(_memberAccess.annotation().referencedDeclaration);
	if (!funcDef)
		throw UnsupportedSolCore(
			"Base-qualified call '" + _memberAccess.memberName() + "' does not reference a function definition.");
	if (!_mostDerived)
		throw UnsupportedSolCore(
			"Base-qualified call to '" + funcDef->name() + "' outside of a contract export context.");

	FunctionDefinition const* target = nullptr;
	bool const isSuper = qualifierType->isSuper();
	if (isSuper)
	{
		// The contract of a super type is the contract lexically containing
		// the call (annotations are assigned once, when that contract was
		// type-checked). Solidity semantics: search _mostDerived's MRO
		// strictly after it.
		ContractDefinition const& definingContract = qualifierType->contractDefinition();
		auto const& hierarchy = _mostDerived->annotation().linearizedBaseContracts;
		if (std::find(hierarchy.begin(), hierarchy.end(), &definingContract) == hierarchy.end())
			throw UnsupportedSolCore(
				"super call in '" + definingContract.name() + "' but '" + definingContract.name()
				+ "' is not a base of '" + _mostDerived->name() + "'.");
		ContractDefinition const* searchStart = definingContract.superContract(*_mostDerived);
		if (!searchStart)
			throw UnsupportedSolCore(
				"super call in '" + definingContract.name() + "' with no remaining base contracts in '"
				+ _mostDerived->name() + "'.");
		target = &funcDef->resolveVirtual(*_mostDerived, searchStart);
	}
	else
		// Explicit `Base.f(...)` is statically bound to the referenced
		// declaration (VirtualLookup::Static) — no virtual lookup.
		target = funcDef;

	if (!target->isOrdinary() || !target->isImplemented())
		throw UnsupportedSolCore(
			"Statically-bound base call to '" + target->name() + "' has no implemented ordinary target.");

	// If the statically-bound target is ALSO the most-derived implementation
	// of its virtual slot, the plain exported slot name denotes exactly this
	// body: emit a plain call instead of aliasing (aliasing would rename the
	// slot's only implementation and orphan every plain-name caller).
	bool needsAlias = &target->resolveVirtual(*_mostDerived) != target;
	return StaticBaseCallTarget{target, needsAlias, isSuper};
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
			"Statically-bound base call target '" + alias + "' was not registered for flattened export.");
	return alias;
}

/// Attach the executable coordinate of a statically-bound base call. The
/// target function is flattened into the current export contract and its
/// already-disambiguated name is the sibling `function` field.
void addStaticBaseCallTarget(Json& _result, StaticBaseCallTarget const& _resolved)
{
	if (!activeExportContract)
		throw UnsupportedSolCore("Statically-bound base call has no active export-contract identity.");
	_result["contractId"] = exportedContractId(*activeExportContract);
	_result["targetKind"] = _resolved.isSuper ? "super" : "base";
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
	if (activeExportContract && _funcDef.isOrdinary() && !_funcDef.name().empty() && _funcDef.virtualSemantics())
		// The resolved slot winner is by construction never alias-registered
		// (resolveStaticBaseCallTarget only flattens targets that are NOT
		// their slot's most-derived implementation), so this lookup yields
		// the plain exported slot name.
		target = &_funcDef.resolveVirtual(*activeExportContract);
	std::string resolvedName = exportedFunctionName(*target);
	if (activeExportContract && !target->isImplemented())
	{
		// The resolved slot winner has no body anywhere in this export
		// unit's own inheritance linearization. Inside an abstract/interface
		// unit this is expected (the definition of an abstract contract):
		// the binding exists only in a concrete deployable's linearization,
		// which that deployable's OWN export resolves and embeds separately.
		// Record it so downstream consumers (the OCaml generator) can
		// distinguish "expected unbound slot" from exporter drift. For a
		// concrete (non-abstract, non-interface) export, solc's type checker
		// guarantees every virtual slot reachable from the deployable
		// resolves to an implementation, so reaching this branch for a
		// concrete contract means exporter/compiler drift: fail closed
		// instead of silently emitting a callee name with no body.
		if (activeExportContract->abstract() || activeExportContract->isInterface())
			unboundVirtualSlotNames.insert(resolvedName);
		else
			throw UnsupportedSolCore(
				"internal virtual call to '" + resolvedName
				+ "' has no implementation in the linearization of concrete contract '" + activeExportContract->name()
				+ "'");
	}
	return resolvedName;
}

std::set<FunctionDefinition const*> collectSuperReferencedFunctions(ContractDefinition const& _contract)
{
	struct SuperReferenceCollector: ASTConstVisitor
	{
		ContractDefinition const& mostDerived;
		std::set<FunctionDefinition const*> functions;

		explicit SuperReferenceCollector(ContractDefinition const& _mostDerived): mostDerived(_mostDerived) {}

		bool visit(FunctionCall const& _call) override
		{
			auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_call.expression());
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

/// The exported name of an event declaration. Every declaration and emit site
/// must have been pre-assigned from the inherited event closure; falling back
/// to a same-named declaration would break the declaration-ID join.
std::string exportedEventName(EventDefinition const& _event)
{
	auto it = exportedEventNames.find(&_event);
	if (it == exportedEventNames.end())
		throw UnsupportedSolCore(
			"Event '" + _event.name()
			+ "' is outside the compiler-resolved event declaration closure.");
	return it->second;
}

/// The true Solidity signature of an event (`E(uint256)`) — the preimage of
/// topic0. Independent of the exported name, which is why disambiguating the
/// name cannot shift a topic hash.
std::string eventSoliditySignature(EventDefinition const& _event)
{
	if (FunctionTypePointer type = _event.functionType(true))
		return type->externalSignature();
	return {};
}

/// Assign exported event names, disambiguating ONLY on collision.
///
/// Non-overloaded events keep their bare source name byte-for-byte, so no
/// existing artifact, consumer or generated Lean helper name changes. An
/// overloaded group gets the same typed-suffix scheme the function overloads
/// already use (`E_uint256` / `E_address`), then the complete assignment is
/// rechecked: if any two events still share a final name — e.g. a contract
/// that declares both `event E(uint256)` and `event E_uint256(...)` — this
/// throws rather than exporting a pair the consumer would silently merge.
void assignExportedEventNames(std::vector<EventDefinition const*> const& _events)
{
	exportedEventNames.clear();

	std::map<std::string, std::vector<EventDefinition const*>> byName;
	for (EventDefinition const* event: _events)
	{
		if (!event)
			continue;
		byName[event->name().empty() ? "_unnamed" : event->name()].push_back(event);
	}

	for (auto const& [rawName, group]: byName)
	{
		if (group.size() <= 1)
		{
			exportedEventNames[group.front()] = rawName;
			continue;
		}
		for (EventDefinition const* event: group)
			exportedEventNames[event] = rawName + "_" + overloadSignatureSuffixOfParameters(event->parameters());
	}

	std::map<std::string, int> finalNameCounts;
	for (auto const& [event, name]: exportedEventNames)
	{
		(void) event;
		++finalNameCounts[name];
	}
	for (auto const& [name, count]: finalNameCounts)
		if (count > 1)
			throw UnsupportedSolCore(
				"exported event name '" + name
				+ "' is shared by more than one event "
				  "declaration after overload disambiguation; the SolCore wire format "
				  "identifies an emitted event by name, so emitting both would silently "
				  "bind every emit site to one of them. Refusing to export this contract.");
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

// A WRITE-PATH-ONLY companion to exportStorageMapLValue: peels a chain of
// plain struct-field `MemberAccess` hops (e.g. `config.targetAmts` above the
// `[k]`) down to a root Identifier bound to a STATE VariableDeclaration,
// building the multi-component path exportDirectAssignment/exportAssignment
// already know how to emit as a `storage_map_set {"path":[...],...}` (the
// wire shape is already fully supported downstream: storage_indexed_root /
// update_storage_path / the narrow-width truncation pass / Summary are all
// path-generic).
//
// This exists because exportStorageMapLValue only resolves a mapping whose
// INDEX ACCESS BASE is itself a bare state-variable Identifier (or a
// namespaced-storage alias) -- `config.targetAmts[k]` has a MemberAccess
// base instead, which is a different (and, before this, unhandled) shape.
// Deliberately NOT wired into the read path (exportExpr's IndexAccess arm):
// reads of this shape already work today via the generic array_get fallback
// (`array_get(field(storage_get(config), targetAmts), k)`), and touching
// that path risks producing a byte-different (even if equally correct)
// artifact for currently-green corpus functions. Kept write-only so it only
// ever ADDS precision (turns a silently-dropped write into an attributed
// one) and never changes anything that already worked.
//
// Every hop must be a storage StructType member; a storage-POINTER or
// parameter root (rather than a state variable) still throws here and falls
// through to the existing local-write path, which is fine: that path is
// coarse but CONTAINED -- the AST write-oracle already marks any function
// whose writes flow through an unresolvable storage-pointer local as
// `unknown`, and Summary/ProofPlan gate all goals for such functions.
std::pair<std::vector<std::string>, Json> exportStorageMapLValueDeep(Expression const& _expr)
{
	auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr);
	if (!indexAccess || !indexAccess->indexExpression())
		throw UnsupportedSolCore("Expected mapping index access.");

	// Peel MemberAccess hops off the index access's base, building the path
	// in leaf-to-root order (reversed below). Each hop's own base must
	// resolve to a storage struct -- namespaced-storage aliases and
	// `_getTokenStorage()`-style getter-prefix patterns are already handled
	// by exportStorageMapLValue above and are deliberately NOT re-matched
	// here, so this function never produces a result that duplicates (or
	// disagrees with) that one.
	std::vector<std::string> reverseHops;
	Expression const* cursor = &indexAccess->baseExpression();
	while (auto const* memberAccess = dynamic_cast<MemberAccess const*>(cursor))
	{
		Type const* hopBaseType = memberAccess->expression().annotation().type;
		if (!hopBaseType || hopBaseType->category() != Type::Category::Struct)
			throw UnsupportedSolCore("Unsupported deep storage-mapping lvalue base.");
		reverseHops.emplace_back(memberAccess->memberName());
		cursor = &memberAccess->expression();
	}
	if (reverseHops.empty())
		// No struct-field hop was peeled at all: this is exactly the shape
		// exportStorageMapLValue already handles (or correctly rejects) --
		// don't duplicate/shadow it.
		throw UnsupportedSolCore("No struct-field hop above this mapping index access.");

	auto const* identifier = dynamic_cast<Identifier const*>(cursor);
	if (!identifier)
		throw UnsupportedSolCore("Deep storage-mapping lvalue root is not an identifier.");
	auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
	if (!decl || !decl->isStateVariable())
		throw UnsupportedSolCore("Deep storage-mapping lvalue root is not a state variable.");

	std::vector<std::string> path;
	path.emplace_back(decl->name());
	for (auto it = reverseHops.rbegin(); it != reverseHops.rend(); ++it)
		path.emplace_back(*it);

	return {path, exportExpr(*indexAccess->indexExpression())};
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
	if (!sourceType || !targetType || sourceType->category() != Type::Category::Integer
		|| targetType->category() != Type::Category::Integer)
		return false;
	auto const* sourceInt = dynamic_cast<IntegerType const*>(sourceType);
	auto const* targetInt = dynamic_cast<IntegerType const*>(targetType);
	if (!sourceInt || !targetInt || targetInt->numBits() >= sourceInt->numBits())
		return false;
	_targetBits = static_cast<int>(targetInt->numBits());
	return true;
}

// --- Internal function used as a value: helpers (continued) ---
//
// See the thread-local state block above (near `unboundVirtualSlotNames`)
// for the overall design. Placed here (after `virtualCallTargetName`,
// `resolveStaticBaseCallTarget`, `exportedFunctionName`,
// `sanitizeExportNameComponent`) and before `exportExpr`'s definition so
// every helper this needs is already fully defined, while `exportExpr`
// itself (used by `lowerInternalCalleeAndArgs` below) only needs its prior
// forward declaration.

/// The (index, declaration) of every parameter of `_fn` whose type is an
/// internal function type — i.e. every parameter bounded defunctionalization
/// can erase from a specialized sibling's signature.
std::vector<std::pair<size_t, VariableDeclaration const*>> bindableFnPtrParams(FunctionDefinition const& _fn)
{
	std::vector<std::pair<size_t, VariableDeclaration const*>> result;
	auto const& params = _fn.parameters();
	for (size_t i = 0; i < params.size(); ++i)
	{
		auto const* fnType = dynamic_cast<FunctionType const*>(params[i]->annotation().type);
		if (fnType && fnType->kind() == FunctionType::Kind::Internal)
			result.emplace_back(i, params[i].get());
	}
	return result;
}

/// True iff none of `_watchedParams` is ever written in `_function`'s body
/// (Assignment LHS — including tuple-assignment components —, `delete`, or
/// referenced at all from inline assembly). This is what PROVES the
/// singleton-target claim rather than assuming it: Solidity function-typed
/// parameters are ordinary mutable locals, so a reassignment mid-body would
/// invalidate a binding computed from the call-site argument.
bool fnPtrParamsNeverWritten(
	FunctionDefinition const& _function, std::set<VariableDeclaration const*> const& _watchedParams)
{
	struct Detector: ASTConstVisitor
	{
		std::set<VariableDeclaration const*> const& watched;
		bool violated = false;
		explicit Detector(std::set<VariableDeclaration const*> const& _w): watched(_w) {}

		void flagIfWatched(Expression const& _expr)
		{
			if (auto const* ident = dynamic_cast<Identifier const*>(&_expr))
				if (auto const* varDecl
					= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					if (watched.count(varDecl))
						violated = true;
		}
		void flagAssignmentTarget(Expression const& _target)
		{
			if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_target))
			{
				for (auto const& component: tuple->components())
					if (component)
						flagAssignmentTarget(*component);
				return;
			}
			flagIfWatched(_target);
		}
		bool visit(Assignment const& _assignment) override
		{
			flagAssignmentTarget(_assignment.leftHandSide());
			return true;
		}
		bool visit(UnaryOperation const& _unary) override
		{
			if (_unary.getOperator() == Token::Delete)
				flagIfWatched(_unary.subExpression());
			return true;
		}
		bool visit(InlineAssembly const& _asm) override
		{
			for (auto const& entry: _asm.annotation().externalReferences)
				if (auto const* varDecl = dynamic_cast<VariableDeclaration const*>(entry.second.declaration))
					if (watched.count(varDecl))
						violated = true;
			return true;
		}
	};

	Detector detector{_watchedParams};
	_function.body().accept(detector);
	return !detector.violated;
}

/// Memoized wrapper around `fnPtrParamsNeverWritten` over ALL of `_function`'s
/// bindable (function-typed) parameters — the admissibility check for
/// specializing `_function` at all, independent of which specific targets a
/// given call site binds.
bool fnPtrCalleeAdmissible(FunctionDefinition const& _function)
{
	auto memoIt = fnPtrCalleeAdmissibleMemo.find(&_function);
	if (memoIt != fnPtrCalleeAdmissibleMemo.end())
		return memoIt->second;
	std::set<VariableDeclaration const*> watched;
	for (auto const& indexAndDecl: bindableFnPtrParams(_function))
		watched.insert(indexAndDecl.second);
	bool admissible = fnPtrParamsNeverWritten(_function, watched);
	fnPtrCalleeAdmissibleMemo[&_function] = admissible;
	return admissible;
}

/// Resolve an argument expression bound to an internal-function-typed
/// parameter to its ONE statically-known target, or nullptr if the shape is
/// anything other than a direct internal-function literal or a pass-through
/// of an already-bound parameter — callers must fail closed on nullptr.
/// Applies the SAME virtual-resolution rule as `virtualCallTargetName`
/// (mirrors solc's own codegen: `FunctionDefinition::resolveVirtual` at the
/// point the pointer value is created).
FunctionDefinition const* resolveFnPtrArgumentTarget(Expression const& _arg)
{
	auto const* ident = dynamic_cast<Identifier const*>(&_arg);
	auto const* member = dynamic_cast<MemberAccess const*>(&_arg);
	Declaration const* referenced = ident	 ? ident->annotation().referencedDeclaration
									: member ? member->annotation().referencedDeclaration
											 : nullptr;
	if (!referenced)
		return nullptr;

	if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(referenced))
	{
		FunctionDefinition const* target = funcDef;
		if (activeExportContract && funcDef->isOrdinary() && !funcDef->name().empty() && funcDef->virtualSemantics())
			target = &funcDef->resolveVirtual(*activeExportContract);
		if (!target->isOrdinary() || !target->isImplemented())
			return nullptr;
		if (target->visibility() == Visibility::External)
			return nullptr;
		return target;
	}

	if (ident)
		if (auto const* varDecl = dynamic_cast<VariableDeclaration const*>(referenced))
		{
			auto bindingIt = activeFnPtrBindings.find(varDecl);
			if (bindingIt != activeFnPtrBindings.end())
				return bindingIt->second->target;
		}

	return nullptr;
}

/// The specialized sibling name for `_baseExportedName` bound per
/// `_bindings` (ascending paramIndex): `<base>` + for each binding
/// `"__fnptr__" + <param label> + "__" + <sanitized target name>`.
std::string
computeFnPtrSpecializedName(std::string const& _baseExportedName, std::vector<FnPtrBinding> const& _bindings)
{
	std::string name = _baseExportedName;
	for (FnPtrBinding const& binding: _bindings)
	{
		std::string paramLabel
			= binding.param->name().empty() ? ("arg" + std::to_string(binding.param->id())) : binding.param->name();
		name += "__fnptr__" + paramLabel + "__" + sanitizeExportNameComponent(binding.targetExportedName);
	}
	return name;
}

/// Register (memoized) a specialization request, enqueueing it for emission
/// the first time it is seen. A name collision between two DIFFERENT
/// (callee, bindings) pairs is an exporter naming-scheme bug — fail closed
/// rather than silently merge two distinct specializations under one name.
std::string registerFnPtrSpecialization(
	FunctionDefinition const& _callee, std::string const& _baseExportedName, std::vector<FnPtrBinding> _bindings)
{
	std::string specializedName = computeFnPtrSpecializedName(_baseExportedName, _bindings);

	auto existingIt = fnPtrSpecializationsByName.find(specializedName);
	if (existingIt != fnPtrSpecializationsByName.end())
	{
		FnPtrSpecializationRequest const& existing = existingIt->second;
		bool same = existing.callee == &_callee && existing.bindings.size() == _bindings.size();
		for (size_t i = 0; same && i < _bindings.size(); ++i)
			same = existing.bindings[i].param == _bindings[i].param
				   && existing.bindings[i].target == _bindings[i].target;
		if (!same)
			throw UnsupportedSolCore(
				"internal-function-value specialization name collision on '" + specializedName
				+ "' between distinct bindings; not modeled.");
		return specializedName;
	}

	FnPtrSpecializationRequest request;
	request.callee = &_callee;
	request.baseExportedName = _baseExportedName;
	request.bindings = std::move(_bindings);
	request.specializedName = specializedName;
	fnPtrSpecializationsByName.emplace(specializedName, std::move(request));
	fnPtrSpecializationQueue.push_back(specializedName);
	return specializedName;
}

/// Applies the SAME predicate `virtualCallTargetName` uses to decide
/// whether a plain-identifier internal call requires virtual resolution,
/// returning the resolved FunctionDefinition (not just its name) so callers
/// can inspect its parameter list. Kept independent of (rather than
/// refactoring) `virtualCallTargetName` to avoid any risk of behavior drift
/// in that already-relied-upon function.
FunctionDefinition const& resolveInternalCallImplementation(FunctionDefinition const& _funcDef)
{
	if (activeExportContract && _funcDef.isOrdinary() && !_funcDef.name().empty() && _funcDef.virtualSemantics())
		return _funcDef.resolveVirtual(*activeExportContract);
	return _funcDef;
}

/// Shared internal-call lowering helper: computes the callee name to emit
/// and the (possibly parameter-erased) argument list for a call to
/// `_resolvedImpl` whose plain (unspecialized) name would be `_plainName`.
/// Returns `{_plainName, args}` UNCHANGED when `_resolvedImpl` has no
/// internal-function-typed parameters — the corpus-wide no-op guarantee for
/// every call site this exporter already knew how to lower.
std::pair<std::string, Json>
lowerInternalCalleeAndArgs(FunctionCall const& _call, FunctionDefinition const& _resolvedImpl, std::string _plainName)
{
	auto exportRootedStorageArgument = [&](Expression const& argument, size_t parameterIndex) -> std::optional<Json>
	{
		if (parameterIndex >= _resolvedImpl.parameters().size())
			return std::nullopt;
		VariableDeclaration const* parameter = _resolvedImpl.parameters()[parameterIndex].get();
		if (isStructuralStorageRefParameter(_resolvedImpl, parameter))
		{
			auto reference = exportOrderedStorageRefValueUse(argument);
			if (!reference)
				reference = exportStorageRefValue(argument);
			if (!reference)
				throw UnsupportedSolCore(
					"Structural storage-reference argument did not resolve to an exact typed path.");
			return reference;
		}
		if (!isStorageRefParameter(parameter))
			return std::nullopt;
		if (hasResidualStorageRefRoot(argument))
		{
			auto reference = exportStorageRefValue(argument);
			if (!reference)
				throw UnsupportedSolCore(
					"Storage-reference local argument did not resolve to an exact typed projection.");
			return reference;
		}
		return exportResolvedStorageRefUse(argument);
	};
	auto exportUnspecializedArgs = [&]()
	{
		auto const& sourceArguments = _call.arguments();
		bool observable = false;
		for (auto const& argument: sourceArguments)
			observable = observable || evalOrderRelevant(*argument);

		auto orderedArguments = sourceArguments;
		bool const named = !_call.names().empty();
		if (named)
			orderedArguments = _call.sortedArguments();

		if (named && observable)
		{
			if (!activeCompilerStack)
				throw UnsupportedSolCore(
					"named internal arguments with observable evaluation order "
					"require an explicit legacy codegen identity.");
			if (activeCompilerStack->viaIR())
				throw UnsupportedSolCore(
					"named internal arguments with observable evaluation order "
					"diverge under via-IR; only the measured legacy parameter-order "
					"row is admitted.");
		}

		if (orderedArguments.size() != _resolvedImpl.parameters().size())
			throw UnsupportedSolCore(
				"Internal call argument count disagrees with its resolved implementation.");

		Json args = Json::array();
		for (size_t parameterIndex = 0; parameterIndex < orderedArguments.size(); ++parameterIndex)
		{
			Expression const* argument = orderedArguments[parameterIndex].get();
			if (auto rootedArgument = exportRootedStorageArgument(*argument, parameterIndex))
			{
				args.emplace_back(std::move(*rootedArgument));
				continue;
			}
			if (!observable)
			{
				args.emplace_back(exportExpr(*argument));
				continue;
			}
			auto sourceIt = std::find_if(
				sourceArguments.begin(),
				sourceArguments.end(),
				[&](auto const& sourceArg) { return sourceArg.get() == argument; });
			solAssert(sourceIt != sourceArguments.end(), "");
			size_t sourceIndex = static_cast<size_t>(std::distance(sourceArguments.begin(), sourceIt));
			std::string label
				= "internal_arg_" + std::to_string(parameterIndex) + "_source_" + std::to_string(sourceIndex);
			VariableDeclaration const* parameter = _resolvedImpl.parameters()[parameterIndex].get();
			if (!parameter || !parameter->annotation().type)
				throw UnsupportedSolCore(
					"Internal call argument lacks a resolved destination parameter type.");
			args.emplace_back(pinExpressionOnce(*argument, label, parameter->annotation().type));
		}
		return args;
	};

	std::vector<std::pair<size_t, VariableDeclaration const*>> bindable = bindableFnPtrParams(_resolvedImpl);
	if (bindable.empty())
		return {std::move(_plainName), exportUnspecializedArgs()};

	if (_call.arguments().size() != _resolvedImpl.parameters().size())
		throw UnsupportedSolCore(
			"call to '" + _plainName
			+ "' has an argument-count mismatch against its resolved "
			  "implementation; cannot bind its internal-function-typed parameter(s).");
	if (!_resolvedImpl.isOrdinary() || !_resolvedImpl.isImplemented())
		throw UnsupportedSolCore(
			"internal-function-value call to '" + _plainName
			+ "' requires an "
			  "implemented ordinary callee body.");

	// Preserve the existing singleton specialization as the zero-runtime-cost
	// fast path. Any parameter whose value is not a direct producer-resolved
	// function declaration (including locals, storage reads, returns, and
	// caller parameters) keeps the ordinary function signature and transports
	// its typed tag to the callee's bounded runtime dispatcher instead.
	bool canSpecialize = _call.names().empty() && fnPtrCalleeAdmissible(_resolvedImpl);
	std::vector<FnPtrBinding> bindings;
	if (canSpecialize)
		for (auto const& [index, param]: bindable)
		{
			FunctionDefinition const* target = resolveFnPtrArgumentTarget(*_call.arguments()[index]);
			if (!target)
			{
				canSpecialize = false;
				break;
			}
			FnPtrBinding binding;
			binding.param = param;
			binding.target = target;
			binding.targetExportedName = exportedFunctionName(*target);
			bindings.push_back(std::move(binding));
		}

	if (!canSpecialize)
		return {std::move(_plainName), exportUnspecializedArgs()};

	std::set<size_t> bindableIndices;
	for (auto const& [index, param]: bindable)
	{
		(void) param;
		bindableIndices.insert(index);
	}
	Json args = Json::array();
	for (size_t i = 0; i < _resolvedImpl.parameters().size(); ++i)
		if (!bindableIndices.count(i))
		{
			if (auto rootedArgument = exportRootedStorageArgument(*_call.arguments()[i], i))
				args.emplace_back(std::move(*rootedArgument));
			else
				args.emplace_back(exportExpr(*_call.arguments()[i]));
		}

	std::string specializedName = registerFnPtrSpecialization(_resolvedImpl, _plainName, std::move(bindings));
	return {std::move(specializedName), std::move(args)};
}

/// Export one attached (`using for`) call from solc's typed AST. The
/// FunctionType flag is the producer-owned proof that the member-access base
/// is the bound first argument; spelling, visibility and arity are not used
/// to rediscover that fact downstream.
std::optional<Json> exportUsingForCall(FunctionCall const& _call, MemberAccess const& _memberAccess)
{
	auto const* functionType = dynamic_cast<FunctionType const*>(_call.expression().annotation().type);
	if (!functionType || !functionType->hasBoundFirstArgument())
		return std::nullopt;

	auto const* function = dynamic_cast<FunctionDefinition const*>(_memberAccess.annotation().referencedDeclaration);
	if (!function)
		return std::nullopt;

	auto const* contract = dynamic_cast<ContractDefinition const*>(function->scope());
	bool const isLibraryFunction = contract && contract->isLibrary();
	bool const isFreeFunction = dynamic_cast<SourceUnit const*>(function->scope());
	if (!isLibraryFunction && !isFreeFunction)
		throw UnsupportedSolCore(
			"Attached call '" + function->name() + "' did not resolve to a library or free function declaration.");
	if (_call.arguments().size() + 1 != function->parameters().size())
		throw UnsupportedSolCore(
			"Attached call '" + function->name()
			+ "' has inconsistent typed receiver provenance: exactly one bound first "
			  "argument was expected.");

	Json result = Json::object();
	result["kind"] = "internal_call";
	result["function"] = isPublicLibraryStructuralStorageFunction(*function) ? storageRefInternalEntryName(*function)
																			 : exportedFunctionName(*function);
	addInternalLibraryCallContractId(result, *function);
	result["args"] = Json::array();
	auto exportBoundArgument = [&](Expression const& argument, size_t parameterIndex) -> Json
	{
		VariableDeclaration const* parameter = function->parameters()[parameterIndex].get();
		if (isStructuralStorageRefParameter(*function, parameter))
		{
			auto reference = exportOrderedStorageRefValueUse(argument);
			if (!reference)
				reference = exportStorageRefValue(argument);
			if (!reference)
				throw UnsupportedSolCore(
					"Attached structural storage-reference receiver did not resolve "
					"to an exact typed path.");
			return std::move(*reference);
		}
		if (isStorageRefParameter(parameter))
		{
			if (hasResidualStorageRefRoot(argument))
			{
				auto reference = exportStorageRefValue(argument);
				if (!reference)
					throw UnsupportedSolCore(
						"Attached storage-reference local argument did not resolve "
						"to an exact typed projection.");
				return std::move(*reference);
			}
			if (auto rootedArgument = exportResolvedStorageRefUse(argument))
				return std::move(*rootedArgument);
		}
		return exportExpr(argument);
	};
	result["args"].emplace_back(exportBoundArgument(_memberAccess.expression(), 0));
	for (size_t i = 0; i < _call.arguments().size(); ++i)
		result["args"].emplace_back(exportBoundArgument(*_call.arguments()[i], i + 1));
	return result;
}

/// RAII guard installing `activeFnPtrBindings` for the duration of exporting
/// one specialized sibling's body, restoring the previous (normally empty —
/// specialized bodies are exported top-level, not nested inside another
/// specialization's export) value on scope exit for exception safety.
struct FnPtrBindingScope
{
	std::map<VariableDeclaration const*, FnPtrBinding const*> saved;
	explicit FnPtrBindingScope(FnPtrSpecializationRequest const& _request): saved(activeFnPtrBindings)
	{
		activeFnPtrBindings.clear();
		for (FnPtrBinding const& binding: _request.bindings)
			activeFnPtrBindings[binding.param] = &binding;
	}
	~FnPtrBindingScope() { activeFnPtrBindings = saved; }
	FnPtrBindingScope(FnPtrBindingScope const&) = delete;
	FnPtrBindingScope& operator=(FnPtrBindingScope const&) = delete;
};

/// Narrow-bytesN width convention (SolCoreBase.ml [BytesW] doc): export
/// `_operand` as a value of `_targetType` when the target is a fixed bytesM
/// and the type checker converted the operand implicitly — a register no-op
/// on the EVM but a REAL adjustment on the right-aligned carrier. Two shapes
/// are materialized (the widening scanner refuses them at sites that do not
/// route through this helper): a string/hex literal RIGHT-pads to M, and a
/// narrower bytesN operand shifts up by 8*(M-N) (untagged full-word shl —
/// exact, since v < 2^(8N)). Compile-time rationals are exact-width or zero
/// by typing, so their numeric export is already correct. Non-fixed-bytes
/// targets export unchanged.
Json exportExprCoercedToFixedBytes(Expression const& _operand, Type const* _targetType)
{
	auto const* targetFB = dynamic_cast<FixedBytesType const*>(_targetType);
	if (!targetFB)
		return exportExpr(_operand);
	Type const* operandType = _operand.annotation().type;
	if (auto const* literal = dynamic_cast<Literal const*>(&_operand))
		if (dynamic_cast<StringLiteralType const*>(operandType))
		{
			canonicalLiteralTargets[literal] = _targetType;
			return exportExpr(_operand);
		}
	if (auto const* operandFB = dynamic_cast<FixedBytesType const*>(operandType))
		if (operandFB->numBytes() < targetFB->numBytes())
		{
			Json shiftLit = Json::object();
			shiftLit["kind"] = "u256";
			shiftLit["value"] = std::to_string(8u * (targetFB->numBytes() - operandFB->numBytes()));
			Json shifted = Json::object();
			shifted["kind"] = "u256_shl";
			shifted["bits"] = 256;
			shifted["lhs"] = exportExpr(_operand);
			shifted["rhs"] = shiftLit;
			return shifted;
		}
	return exportExpr(_operand);
}

Json exportExpr(Expression const& _expr)
{
	if (activeHoistScope)
	{
		auto memoIt = activeHoistScope->memo.find(static_cast<int64_t>(_expr.id()));
		if (memoIt != activeHoistScope->memo.end())
			return localExpr(memoIt->second);
	}

	if (!exportingStorageRefValueExpression && isStorageReferenceType(_expr.annotation().type)
		&& hasResidualStorageRefRoot(_expr))
	{
		StorageRefValueExpressionScope scope;
		auto reference = exportStorageRefValue(_expr);
		if (!reference)
			throw UnsupportedSolCore("Residual storage-reference read did not resolve to an exact typed path.");
		return storageRefGetJson(std::move(*reference), _expr.annotation().type);
	}

	// A declaration reference whose compiler-resolved VALUE type is internal
	// function is the sole nonzero tag minting site. Locals, parameters,
	// storage reads, conditionals and returns merely transport this value.
	if (auto const* fnType = dynamic_cast<FunctionType const*>(_expr.annotation().type))
		if (fnType->kind() == FunctionType::Kind::Internal)
		{
			Declaration const* referenced = nullptr;
			if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
				referenced = identifier->annotation().referencedDeclaration;
			else if (auto const* member = dynamic_cast<MemberAccess const*>(&_expr))
				referenced = member->annotation().referencedDeclaration;
			if (auto const* declaration = dynamic_cast<FunctionDefinition const*>(referenced))
			{
				FunctionDefinition const* target = resolveFnPtrArgumentTarget(_expr);
				if (!target)
					throw UnsupportedSolCore("Internal function value did not resolve to one implemented candidate.");
				Json wireType = exportResolvedType(fnType);
				InternalFnCandidateRecord const& candidate
					= registerInternalFnCandidate(*fnType, wireType, *declaration, *target);
				Json result = Json::object();
				result["kind"] = "internal_fn_ref";
				result["table"] = internalFnTablesByFingerprint.at(fnType->richIdentifier()).tableId;
				result["tag"] = candidate.tag;
				result["function"] = candidate.functionName;
				result["declarationContractId"] = candidate.declarationContractId;
				result["targetContractId"] = candidate.targetContractId;
				result["type"] = wireType;
				return result;
			}
		}

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
		if (auto const* rationalType = dynamic_cast<RationalNumberType const*>(literal->annotation().type))
		{
			if (rationalType->isFractional())
				throw UnsupportedSolCore("Fractional numeric literal has no exact u256 lowering.");
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = rationalType->literalValue(literal).str();
			return result;
		}
		// Address literals: 0x... with 40 hex digits.
		if (literal->annotation().type && literal->annotation().type->category() == Type::Category::Address)
		{
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = literal->value();
			return result;
		}
		if (literal->token() == Token::StringLiteral || literal->token() == Token::UnicodeStringLiteral
			|| literal->token() == Token::HexStringLiteral)
		{
			std::string encoding;
			switch (literal->token())
			{
			case Token::StringLiteral:
				encoding = "plain";
				break;
			case Token::UnicodeStringLiteral:
				encoding = "unicode";
				break;
			case Token::HexStringLiteral:
				encoding = "hex";
				break;
			default:
				throw UnsupportedSolCore("Unknown byte/string literal encoding.");
			}

			Type const* targetType = nullptr;
			if (auto target = canonicalLiteralTargets.find(literal); target != canonicalLiteralTargets.end())
				targetType = target->second;

			Json target = Json::object();
			if (auto const* fixed = dynamic_cast<FixedBytesType const*>(targetType))
			{
				if (literal->value().size() > fixed->numBytes())
					throw UnsupportedSolCore("Byte/string literal exceeds its declared bytesN target width.");
				target["kind"] = "fixed_bytes";
				target["width"] = fixed->numBytes();
			}
			else if (auto const* array = dynamic_cast<ArrayType const*>(targetType))
			{
				if (array->isString())
					target["kind"] = "string";
				else if (array->isByteArray())
					target["kind"] = "bytes";
				else
					throw UnsupportedSolCore("Byte/string literal has a non-byte-array target.");
			}
			else if (!targetType || dynamic_cast<StringLiteralType const*>(targetType))
				target["kind"] = literal->token() == Token::HexStringLiteral ? "bytes" : "string";
			else
				throw UnsupportedSolCore("Byte/string literal has an incompatible declared target.");

			static char const digits[] = "0123456789abcdef";
			std::string const& raw = literal->value();
			std::string sourceBytes;
			sourceBytes.reserve(raw.size() * 2);
			for (char c: raw)
			{
				sourceBytes += digits[static_cast<unsigned char>(c) >> 4];
				sourceBytes += digits[static_cast<unsigned char>(c) & 0xf];
			}

			Json result = Json::object();
			result["kind"] = "typed_literal";
			result["encoding"] = encoding;
			result["sourceBytes"] = sourceBytes;
			result["width"] = raw.size();
			result["target"] = target;
			return result;
		}
		throw UnsupportedSolCore("Only boolean, numeric, address, and typed byte/string literals are supported.");
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
			// Immutable values are deployment-time state, never initializer
			// substitutions or persistent-storage fields.
			if (decl->immutable())
				return immutableGet(*decl);
			// Constants retain their source-level initializer substitution.
			static thread_local int inlineDepth = 0;
			if (decl->isConstant() && decl->value() && inlineDepth < 3)
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
							(void) literal;
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
			// EIP-1153 `transient` state variable read: a dedicated node,
			// NEVER storage_get — the value lives in the transaction-scoped
			// transient slot space (ExecState.transient runtime-side), not
			// in the persistent Storage record. The crate-level
			// `transient_state` list (exportStateVars) carries the
			// name -> slot binding this node references by name.
			if (isTransientStateVar(decl))
			{
				result["kind"] = "transient_get";
				result["field"] = decl->name();
				return result;
			}
			if (decl->isStateVariable())
			{
				result["kind"] = "storage_get";
				result["field"] = decl->name();
				return result;
			}
			if (storageRefValueLocals.count(decl))
			{
				Json reference = Json::object();
				reference["kind"] = "local";
				reference["name"] = decl->name().empty() ? identifier->name() : decl->name();
				return storageRefGetJson(std::move(reference), decl->annotation().type);
			}
			// General storage-reference-variable alias tracking (Phase 1a
			// §3.3): a tracked, resolved local storage-ref alias reads as a
			// LIVE re-projection of current storage, not a bind-time value
			// copy — this is what lets every downstream MemberAccess/
			// IndexAccess wrapping this read (handled generically elsewhere
			// in exportExpr) produce the same rooted shape a direct,
			// non-aliased chain would.
			{
				auto aliasIt = storageRefAliasTargets.find(decl);
				if (aliasIt != storageRefAliasTargets.end())
					return aliasReadJson(aliasIt->second);
			}
			result["kind"] = "local";
			result["name"] = decl->name().empty() ? identifier->name() : decl->name();
			return result;
		}
		// Handle magic variables like "this" — they are not locals
		if (auto const* magicDecl
			= dynamic_cast<MagicVariableDeclaration const*>(identifier->annotation().referencedDeclaration))
		{
			(void) magicDecl;
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
		// (internal function pointers) in any NON-ARGUMENT value context
		// (assignment RHS, return, storage store, comparison, ...). A direct
		// internal-function literal passed as a call ARGUMENT into an
		// internal-function-typed parameter is handled separately by
		// bounded defunctionalization (see lowerInternalCalleeAndArgs,
		// above exportExpr) and never reaches this fallback; every other
		// use of a function value remains unmodeled.
		if (dynamic_cast<FunctionDefinition const*>(identifier->annotation().referencedDeclaration))
			throw UnsupportedSolCore(
				"An internal function used as a value (a function pointer) "
				"outside of a direct call-site argument is not modeled by the "
				"SolCore exporter.");
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
				"'" + identifier->name()
				+ "' is a declaration-level reference (not a variable) used "
				  "in value context, which the SolCore exporter cannot "
				  "represent as a real value.");
		// No declaration resolved at all for this identifier. Every genuine
		// local-variable reference resolves to a VariableDeclaration above,
		// so reaching here means name resolution didn't attach a
		// declaration -- guessing "local" would silently synthesize a
		// reference to a variable that may never have been declared in the
		// translated scope. Fail closed rather than guess.
		throw UnsupportedSolCore(
			"Identifier '" + identifier->name()
			+ "' has no resolved declaration; the SolCore exporter cannot "
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
			auto const* decl
				= dynamic_cast<MagicVariableDeclaration const*>(baseIdentifier->annotation().referencedDeclaration);
			if (decl)
			{
				// `this` is a MagicVariableDeclaration, but its members are
				// this contract's own FUNCTIONS, not CallEnv fields:
				// `this.foo` (in `abi.encodeCall(this.foo, ...)`, or any
				// external-function-value position) must reach the
				// `.selector`/`.address`/external-function-member arms below,
				// which lower it from the declaration solc already resolved.
				// Routing it through runtimeFieldForMagicMember instead threw
				// "Unsupported magic member access: this.foo" before those
				// arms ever ran. Every OTHER magic base (`msg`, `block`,
				// `tx`) keeps the loud throw for an unmapped member, and a
				// `this.<member>` that matches no arm below still fails
				// closed at the end of exportExpr.
				if (baseIdentifier->name() != "this")
				{
					Json result = Json::object();
					result["kind"] = "state_get";
					result["path"] = jsonStringArray(
						{"env", runtimeFieldForMagicMember(baseIdentifier->name(), memberAccess->memberName())});
					return result;
				}
			}
		}

		// ContractName.stateVar access — accessing state variable through contract type qualifier
		if (memberAccess->annotation().referencedDeclaration)
		{
			if (auto const* varDecl
				= dynamic_cast<VariableDeclaration const*>(memberAccess->annotation().referencedDeclaration))
			{
				if (varDecl->isStateVariable())
				{
					// Constant state variables retain initializer substitution.
					if (varDecl->isConstant() && varDecl->value())
						return exportExpr(*varDecl->value());
					if (varDecl->immutable())
						return immutableGet(*varDecl);
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
				// Every balance read uses the shared per-address map, including
				// `address(this).balance`. The current address is exported as
				// `env.thisAddress`, so the consumer reads the same canonical
				// row that CALL and CREATE debit.
				Json result = Json::object();
				result["kind"] = "balance_of";
				result["address"] = exportExpr(memberAccess->expression());
				return result;
			}
		}

		// `<address>.codehash` (EIP-1052 EXTCODEHASH). A per-address read of
		// the shared world's code-hash mapping (`WorldState.codeHash`, a
		// defaulted runtime field in the same style as `balances`), mirroring
		// the `.balance`/`.code.length` world-read shapes above. The value is
		// UNCONSTRAINED environment data: EIP-1052's zero-for-nonexistent
		// vs keccak256("")-for-empty distinction is a property of the world
		// the frame runs against, not a rule this exporter states, so nothing
		// here relates a code hash to a code size or to emptiness.
		if (memberAccess->memberName() == "codehash")
		{
			Type const* codehashBaseType = memberAccess->expression().annotation().type;
			if (codehashBaseType && codehashBaseType->category() == Type::Category::Address)
			{
				Json result = Json::object();
				result["kind"] = "extcodehash";
				result["address"] = exportExpr(memberAccess->expression());
				return result;
			}
		}

		// Array .length access
		if (memberAccess->memberName() == "length")
		{
			if (auto const* codeAccess = dynamic_cast<MemberAccess const*>(&memberAccess->expression()))
			{
				Type const* codeBaseType = codeAccess->expression().annotation().type;
				if (codeAccess->memberName() == "code" && codeBaseType
					&& codeBaseType->category() == Type::Category::Address)
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
					if (typeArg && typeArg->category() == Type::Category::Contract)
					{
						auto const* contractType = dynamic_cast<ContractType const*>(typeArg);
						if (!contractType)
							throw UnsupportedSolCore("Contract metatype member has no resolved contract definition.");
						ContractDefinition const& contract = contractType->contractDefinition();
						if (memberAccess->memberName() == "name")
							return canonicalStringTypedLiteral(contract.name());
						if (memberAccess->memberName() == "creationCode" || memberAccess->memberName() == "runtimeCode")
						{
							Json result = Json::object();
							result["kind"] = "contract_code";
							result["which"] = memberAccess->memberName() == "creationCode" ? "creation" : "runtime";
							result["contract"] = exportedContractId(contract);
							return result;
						}
					}
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
									// `i256` literals carry the canonical
									// two's-complement WORD (digits-only,
									// < 2^256), never a signed decimal --
									// `IntegerType::min()`/`max()` already
									// apply `s2u` for signed types (see
									// Types.cpp), so use those instead of
									// the raw signed `minValue()`/
									// `maxValue()` bigints.
									result["kind"] = "i256";
									if (memberAccess->memberName() == "min")
										result["value"] = intType->min().str();
									else
										result["value"] = intType->max().str();
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
									memberAccess->memberName() == "min" ? enumType->minValue() : enumType->maxValue());
								return result;
							}
						}
						// Genuinely unknown/unhandled type(X).max/min type argument --
						// fail closed instead of silently substituting a `0` that a
						// caller (e.g. a `require(x <= type(X).max)` bound check) would
						// then treat as a real, meaningful value.
						throw UnsupportedSolCore(
							"type(X)." + memberAccess->memberName()
							+ " is not supported by the SolCore exporter for this type "
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
					if (typeArg && typeArg->category() == Type::Category::Contract)
						throw UnsupportedSolCore(
							"Unsupported contract metatype member '" + memberAccess->memberName() + "'.");
				}
			}
		}

		// Bytes .length on dynamic bytes
		if (memberAccess->memberName() == "length")
		{
			Type const* baseType = memberAccess->expression().annotation().type;
			if (baseType
				&& (baseType->category() == Type::Category::Array
					|| baseType->category() == Type::Category::FixedBytes))
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
			FunctionType const* funType = dynamic_cast<FunctionType const*>(selBase.annotation().type);
			FunctionDefinition const* funcDef = nullptr;
			std::string methodName;
			if (auto const* baseMember = dynamic_cast<MemberAccess const*>(&selBase))
			{
				methodName = baseMember->memberName();
				funcDef = dynamic_cast<FunctionDefinition const*>(baseMember->annotation().referencedDeclaration);
			}
			if (!funcDef)
				if (auto const* baseIdent = dynamic_cast<Identifier const*>(&selBase))
					funcDef = dynamic_cast<FunctionDefinition const*>(baseIdent->annotation().referencedDeclaration);
			if (methodName.empty() && funcDef)
				methodName = funcDef->name();

			// `.selector` of an external function-pointer VALUE (storage var,
			// local, parameter, try-binding): no FunctionDefinition resolves
			// and the FunctionType has no declaration, so the static
			// selector-constant path below (externalIdentifierHex, which
			// solAsserts a declaration) cannot apply. The selector is a
			// runtime projection of the 24-byte (address, selector) value.
			if (funType && funType->kind() == FunctionType::Kind::External && !funcDef && !funType->hasDeclaration())
			{
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = "extfn_selector";
				result["args"] = Json::array();
				result["args"].emplace_back(exportExpr(selBase));
				return result;
			}

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
					if (auto const* baseIdent = dynamic_cast<Identifier const*>(&baseMember->expression()))
					{
						if (auto const* cd
							= dynamic_cast<ContractDefinition const*>(baseIdent->annotation().referencedDeclaration))
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
		// `.address` of an external function-pointer expression: a runtime
		// projection of the (address, selector) value. This also covers the
		// bound-member form `I(target).echo.address` — the base exports as an
		// extfn_pack of the receiver (see the value-use arm below), and the
		// projection composes over it.
		if (memberAccess->memberName() == "address")
		{
			auto const* addrFnType = dynamic_cast<FunctionType const*>(memberAccess->expression().annotation().type);
			if (addrFnType && addrFnType->kind() == FunctionType::Kind::External)
			{
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = "extfn_address";
				result["args"] = Json::array();
				result["args"].emplace_back(exportExpr(memberAccess->expression()));
				return result;
			}
		}

		// VALUE use of a statically-declared external function member
		// (`I(target).echo`, `this.value` outside a call position): construct
		// the runtime (address, selector) pair from the receiver's address
		// and the selector solc itself resolved from the declaration. Only
		// fires when the member resolves to a FunctionDefinition with kind
		// External — call positions never reach exportExpr on the
		// MemberAccess itself (FunctionCall arms match first), and
		// namespace accesses (`I.echo`, Kind::Declaration) do not match.
		{
			auto const* valFnType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type);
			auto const* valFnDef
				= dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration);
			if (valFnType && valFnDef && valFnType->kind() == FunctionType::Kind::External)
			{
				std::string selectorHex = valFnType->hasDeclaration() ? valFnType->externalIdentifierHex()
																	  : valFnDef->externalIdentifierHex();
				if (selectorHex.empty())
					throw UnsupportedSolCore(
						"external function member value without a resolvable selector: " + memberAccess->memberName());
				Json sel = Json::object();
				sel["kind"] = "u256";
				sel["value"] = "0x" + selectorHex;
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = "extfn_pack";
				result["args"] = Json::array();
				result["args"].emplace_back(exportExpr(memberAccess->expression()));
				result["args"].emplace_back(std::move(sel));
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
		// `x[i]` on a FIXED bytesN base is byte EXTRACTION from a word
		// value, not an array read: the old `array_get` lowering here
		// produced Lean that applied a list read to a U256 (ill-typed, so
		// it could never elaborate). Under the narrow-bytesN width
		// convention (right-aligned carrier) the i-th byte FROM THE LEFT is
		// `(v >> 8*(N-1-i)) & 0xff` with a Panic 0x32 bounds check — the
		// dedicated width-mangled runtime helper the generator emits
		// (bytesn-ident-index pin: in-range yields the byte as `bytes1`,
		// out-of-range panics 0x32 exactly like an array read).
		if (baseType && baseType->category() == Type::Category::FixedBytes)
		{
			auto const* fbType = dynamic_cast<FixedBytesType const*>(baseType);
			Json result = Json::object();
			result["kind"] = "internal_call";
			result["function"] = "bytesn_get__" + std::to_string(fbType->numBytes());
			result["args"] = Json::array();
			result["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
			result["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
			return result;
		}
		if (baseType && baseType->category() == Type::Category::Array)
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

		// User-defined operator (UDVT `using {f as +} for T global`): the
		// type checker statically bound this operation to a (file-level,
		// pure) free-function definition. The built-in word-level arms below
		// would silently compute the WRONG value (spechunt finding G1: a
		// fixed-point `*` without the 1e18 rescale) and the wrong checked
		// semantics (unsigned u256_* on a signed underlying word). Dispatch
		// to the bound definition exactly like an ordinary internal call —
		// static resolution by solc's own type checker, no name guessing;
		// the bucket-1 out-of-linearization closure embeds the definition.
		if (binary->annotation().userDefinedFunction.set() && *binary->annotation().userDefinedFunction)
		{
			FunctionDefinition const* opFn = *binary->annotation().userDefinedFunction;
			Json call = Json::object();
			call["kind"] = "internal_call";
			call["function"] = exportedFunctionName(*opFn);
			call["args"] = Json::array();
			call["args"].emplace_back(exportExpr(binary->leftExpression()));
			call["args"].emplace_back(exportExpr(binary->rightExpression()));
			return call;
		}
		// Built-in short-circuit operators with an order-relevant RHS need a
		// statement-level result temp whose RHS evaluation lives only inside
		// the selected branch. Pure RHS expressions retain the compact wire
		// node below.
		if ((binary->getOperator() == Token::And || binary->getOperator() == Token::Or)
			&& evalOrderRelevant(binary->rightExpression()))
		{
			if (!activeHoistScope || !activeHoistScope->allowed)
				throw UnsupportedSolCore(
					"short-circuit RHS effects require a once-evaluated "
					"statement context.");
			HoistScope* parent = activeHoistScope;
			auto memoIt = parent->memo.find(static_cast<int64_t>(binary->id()));
			if (memoIt != parent->memo.end())
				return localExpr(memoIt->second);
			Json lhs = pinExpressionOnce(binary->leftExpression(), "short_circuit_lhs");
			std::string resultName
				= "__solcore_evalorder_short_circuit_result_" + std::to_string(stableSyntheticNodeId(*binary));

			Json init = Json::object();
			init["kind"] = "bool";
			init["value"] = binary->getOperator() == Token::Or;
			Json resultLet = Json::object();
			resultLet["kind"] = "let";
			resultLet["sourceDeclarationId"] = Json();
			resultLet["name"] = resultName;
			resultLet["type"] = "bool";
			resultLet["value"] = std::move(init);
			parent->statements.emplace_back(std::move(resultLet));

			Json rhs;
			std::vector<Json> rhsStatements;
			{
				HoistScopeGuard rhsScope({&binary->rightExpression()});
				rhs = exportExpr(binary->rightExpression());
				rhsStatements = rhsScope.takeStatements();
			}
			rhsStatements.emplace_back(localAssignment(resultName, std::move(rhs)));
			std::vector<Json> constantBranch;
			Json constant = Json::object();
			constant["kind"] = "bool";
			constant["value"] = binary->getOperator() == Token::Or;
			constantBranch.emplace_back(localAssignment(resultName, std::move(constant)));

			Json branch = Json::object();
			branch["kind"] = "if";
			branch["cond"] = std::move(lhs);
			if (binary->getOperator() == Token::And)
			{
				branch["then"] = blockFromStatements(std::move(rhsStatements));
				branch["else"] = blockFromStatements(std::move(constantBranch));
			}
			else
			{
				branch["then"] = blockFromStatements(std::move(constantBranch));
				branch["else"] = blockFromStatements(std::move(rhsStatements));
			}
			parent->statements.emplace_back(std::move(branch));
			parent->memo.emplace(static_cast<int64_t>(binary->id()), resultName);
			return localExpr(resultName);
		}

		// `==` / `!=` on FUNCTION-typed operands. External function values
		// compare as the masked (address, selector) pair (solc's
		// externalFunctionPointersEqualFunction); route to the dedicated
		// runtime helper — the generic u256_eq arm below would compare two
		// record values as words, which is ill-typed under the dedicated
		// external_function SolCore type. Internal function values compare
		// their producer-minted 8-byte tags; equal tags identify the same
		// resolved candidate in the one closed table for this exact type.
		if ((binary->getOperator() == Token::Equal || binary->getOperator() == Token::NotEqual)
			&& binary->annotation().commonType
			&& binary->annotation().commonType->category() == Type::Category::Function)
		{
			auto const* fnCommon = dynamic_cast<FunctionType const*>(binary->annotation().commonType);
			if (!fnCommon)
				throw UnsupportedSolCore("Function equality has no compiler-resolved FunctionType.");
			char const* helper = nullptr;
			if (fnCommon->kind() == FunctionType::Kind::External)
				helper = binary->getOperator() == Token::Equal ? "extfn_eq" : "extfn_ne";
			else if (fnCommon->kind() == FunctionType::Kind::Internal)
				helper = binary->getOperator() == Token::Equal ? "intfn_eq" : "intfn_ne";
			else
				throw UnsupportedSolCore("Equality is supported only for internal or external function values.");
			Json call = Json::object();
			call["kind"] = "internal_call";
			call["function"] = helper;
			call["args"] = Json::array();
			call["args"].emplace_back(exportExpr(binary->leftExpression()));
			call["args"].emplace_back(exportExpr(binary->rightExpression()));
			return call;
		}

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
		// SolCore design "Signed integer comparison, shift, and division
		// primitives" landed the signed counterparts
		// (`i256_lt`/`le`/`gt`/`ge`/`div`/`mod`/`sar`, plus the
		// declared-width `i256_add`/`sub`/`mul`/`neg` family and
		// `signextend`) that the OCaml frontend and Lean runtime now
		// consume end-to-end: every one of these operators routes to its
		// signed counterpart when the operand type is a signed integer,
		// closing the fail-open gap this comment used to describe. Signed
		// `**` (`Token::Exp`) and narrow signed `<<` still have no
		// faithful lowering and remain fail-closed (see their cases
		// below).
		auto isSignedIntegerType = [](Type const* _type) -> bool
		{
			if (!_type || _type->category() != Type::Category::Integer)
				return false;
			auto const* intType = dynamic_cast<IntegerType const*>(_type);
			return intType && intType->isSigned();
		};

		switch (binary->getOperator())
		{
		case Token::Add:
			if (auto bits = signedOperationBits(binary->annotation().commonType))
			{
				result["kind"] = "i256_add";
				result["bits"] = static_cast<int>(*bits);
				markUncheckedContext(result);
			}
			else
			{
				result["kind"] = "u256_add";
				markUncheckedContext(result);
				tagUnsignedArithWidth(result, binary->annotation().commonType);
			}
			break;
		case Token::Sub:
			if (auto bits = signedOperationBits(binary->annotation().commonType))
			{
				result["kind"] = "i256_sub";
				result["bits"] = static_cast<int>(*bits);
				markUncheckedContext(result);
			}
			else
			{
				result["kind"] = "u256_sub";
				markUncheckedContext(result);
				tagUnsignedArithWidth(result, binary->annotation().commonType);
			}
			break;
		case Token::Mul:
			if (auto bits = signedOperationBits(binary->annotation().commonType))
			{
				result["kind"] = "i256_mul";
				result["bits"] = static_cast<int>(*bits);
				markUncheckedContext(result);
			}
			else
			{
				result["kind"] = "u256_mul";
				markUncheckedContext(result);
				tagUnsignedArithWidth(result, binary->annotation().commonType);
			}
			break;
		case Token::Div:
			if (auto bits = signedOperationBits(binary->annotation().commonType))
			{
				result["kind"] = "i256_div";
				result["bits"] = static_cast<int>(*bits);
				// The unchecked flag matters here: SDIV(min, -1) reverts in
				// checked context but wraps to `min` in `unchecked { }`.
				markUncheckedContext(result);
			}
			else
				result["kind"] = "u256_div";
			break;
		case Token::Mod:
			if (auto bits = signedOperationBits(binary->annotation().commonType))
			{
				result["kind"] = "i256_mod";
				result["bits"] = static_cast<int>(*bits);
				// SMOD has no overflow case (min tmod -1 = 0), so checked and
				// unchecked coincide: emit one op, ignore `unchecked { }`.
			}
			else
				result["kind"] = "u256_mod";
			break;
		case Token::Exp:
			// `**` is typed at the BASE (left operand) type only — solc's
			// own warning: "The result type of the exponentiation operation
			// is equal to the type of the first operand" — so `commonType`
			// IS the base type and the only sound width source (a literal
			// base with a runtime exponent is typed uint256/int256, i.e.
			// full-word). Checked `**` reverts (Panic 0x11) exactly when
			// the mathematical power leaves the base type's range
			// (checked_exp_unsigned/checked_exp_signed are mathematically
			// exact range checks, incl. the literal-base upperbound
			// helper); `unchecked { ** }` wraps at the base width
			// (wrapping_exp = cleanup(exp(...))). Signed bases route to
			// the dedicated i256_exp kind (negative bases alternate the
			// result's sign with the exponent's parity); the exponent is
			// always an unsigned type per the Solidity type checker.
			// Pinned in plans/spechunt-feature-work-design.md §0.1/§0.2.
			if (auto bits = signedOperationBits(binary->annotation().commonType))
			{
				result["kind"] = "i256_exp";
				result["bits"] = static_cast<int>(*bits);
				markUncheckedContext(result);
			}
			else
			{
				result["kind"] = "u256_exp";
				markUncheckedContext(result);
				tagUnsignedArithWidth(result, binary->annotation().commonType);
			}
			break;
		case Token::Equal:
			result["kind"] = "u256_eq";
			break;
		case Token::NotEqual:
			result["kind"] = "u256_ne";
			break;
		case Token::LessThan:
			result["kind"] = isSignedIntegerType(binary->annotation().commonType) ? "i256_lt" : "u256_lt";
			break;
		case Token::LessThanOrEqual:
			result["kind"] = isSignedIntegerType(binary->annotation().commonType) ? "i256_le" : "u256_le";
			break;
		case Token::GreaterThan:
			result["kind"] = isSignedIntegerType(binary->annotation().commonType) ? "i256_gt" : "u256_gt";
			break;
		case Token::GreaterThanOrEqual:
			result["kind"] = isSignedIntegerType(binary->annotation().commonType) ? "i256_ge" : "u256_ge";
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
			// `x << k` is computed AT THE LHS TYPE'S WIDTH with a
			// truncating cleanup (`and(…, 2^N-1)` for uintN,
			// `signextend(N/8-1, …)` for intN) and NO overflow check even
			// in checked blocks — solc's shift_left_t_<T> is
			// `cleanup_t_T(shl(k, cleanup_t_T(x)))`. For shifts
			// `commonType` is the left operand's type (the shift amount is
			// always unsigned and does not participate), so it is the
			// width source here, exactly as for `**` above. int256 `<<`
			// and full-word unsigned `<<` keep the plain u256_shl (raw
			// mod-2^256 shift is congruence-correct at the full word);
			// narrow signed routes to the dedicated i256_shl kind, narrow
			// unsigned carries its width on u256_shl (both were
			// mis-modeled before: signed threw, unsigned silently skipped
			// the operand-width truncation — narrow-bitwise-cleanup F1).
			// bytesN `<<` re-cleans to the byte lane exactly like a narrow
			// unsigned shift: on the right-aligned bytesN carrier
			// (SolCoreBase.ml [BytesW] doc) solc's
			// `cleanup_t_bytesN(shl(k, x))` is precisely the mod-2^(8N)
			// wrap the width-tagged u256_shl performs, so tag bits = 8N
			// (fixedbytes-shl-eq / fixedbytes-shl-shr-mask pins). `>>` on
			// bytesN needs no tag: flooring the carrier IS the EVM
			// shr+cleanup composite.
			if (auto bits = signedOperationBits(binary->annotation().commonType))
			{
				if (*bits < 256)
				{
					result["kind"] = "i256_shl";
					result["bits"] = static_cast<int>(*bits);
				}
				else
				{
					result["kind"] = "u256_shl";
					result["bits"] = 256;
				}
			}
			else
			{
				result["kind"] = "u256_shl";
				tagUnsignedArithWidth(result, binary->annotation().commonType);
			}
			break;
		case Token::SAR:
			result["kind"] = isSignedIntegerType(binary->leftExpression().annotation().type) ? "i256_sar" : "u256_shr";
			break;
		default:
			throw UnsupportedSolCore("Unsupported binary operator in SolCore exporter.");
		}
		// Narrow-bytesN width convention (SolCoreBase.ml [BytesW] doc): when
		// the operation's commonType is a fixed bytesM, the type checker has
		// implicitly converted each operand to bytesM — a register no-op on
		// the EVM but a REAL adjustment on the right-aligned carrier. Two
		// shapes are materialized here (the widening scanner refuses them
		// elsewhere): a string/hex literal RIGHT-pads to M (bytesn-eq-literal
		// pin: `x != hex"1234"` on bytes4 compares against 0x12340000, not
		// 0x1234), and a narrower bytesN operand shifts up by 8*(M-N)
		// (untagged full-word shl — exact, since v < 2^(8N)). Compile-time
		// rationals are exact-width or zero by typing, so their numeric
		// export is already correct. The shift AMOUNT operand of `<<`/`>>`
		// never participates in the commonType and is exported untouched.
		auto exportBinaryOperandAtCommonType = [&](Expression const& _operand) -> Json
		{ return exportExprCoercedToFixedBytes(_operand, binary->annotation().commonType); };
		bool const rhsIsShiftAmount = binary->getOperator() == Token::SHL || binary->getOperator() == Token::SHR
									  || binary->getOperator() == Token::SAR;
		result["lhs"] = exportBinaryOperandAtCommonType(binary->leftExpression());
		result["rhs"] = rhsIsShiftAmount ? exportExpr(binary->rightExpression())
										 : exportBinaryOperandAtCommonType(binary->rightExpression());
		return result;
	}

	if (auto const* unary = dynamic_cast<UnaryOperation const*>(&_expr))
	{
		Json result = Json::object();

		// User-defined unary operator (UDVT `using {f as -} for T global`):
		// same statically-bound dispatch as the binary case above. Only `-`
		// and `~` are user-bindable unary operators; `++`/`--`/`delete` never
		// carry a userDefinedFunction annotation, so the mutation-hoisting
		// arms below are unaffected.
		if (unary->annotation().userDefinedFunction.set() && *unary->annotation().userDefinedFunction)
		{
			FunctionDefinition const* opFn = *unary->annotation().userDefinedFunction;
			Json call = Json::object();
			call["kind"] = "internal_call";
			call["function"] = exportedFunctionName(*opFn);
			call["args"] = Json::array();
			call["args"].emplace_back(exportExpr(unary->subExpression()));
			return call;
		}

		switch (unary->getOperator())
		{
		case Token::Not:
			result["kind"] = "bool_not";
			result["operand"] = exportExpr(unary->subExpression());
			return result;
		case Token::BitNot:
			// `~x` on a narrow bytesN must re-clean to the byte lane: the
			// full-word bitnot would set every bit above 2^(8N), breaking
			// the right-aligned carrier invariant (SolCoreBase.ml [BytesW]
			// doc) and hence every comparison on the result. On the
			// carrier, lane-not is exactly `x XOR (2^(8N)-1)` — lowered
			// with existing width-less nodes, no new IR kind needed.
			if (auto const* fbType = dynamic_cast<FixedBytesType const*>(unary->annotation().type))
				if (fbType->numBytes() < 32)
				{
					u256 const mask = (u256(1) << (8 * fbType->numBytes())) - 1;
					Json maskLit = Json::object();
					maskLit["kind"] = "u256";
					maskLit["value"] = mask.str();
					result["kind"] = "u256_bitxor";
					result["lhs"] = exportExpr(unary->subExpression());
					result["rhs"] = maskLit;
					return result;
				}
			result["kind"] = "u256_bitnot";
			result["operand"] = exportExpr(unary->subExpression());
			return result;
		case Token::Inc:
		case Token::Dec:
			// `x++`/`x--`/`++x`/`--x` used as a SUB-expression. The old
			// lowering here computed only a pure value (and even the wrong
			// one for postfix: `x+1` instead of the pre-value) and silently
			// dropped the write — a confirmed soundness bug (e.g.
			// `abi.encode(nonces[owner]++, ...)` in permit lost the nonce
			// bump entirely). Hoist the real mutation to statement level
			// and substitute the correctly-captured temp value instead.
			return exportHoistedUnaryMutation(*unary);
		case Token::Sub:
		{
			// Per >=0.8 typing, unary minus is only well-typed on signed
			// integers or on a compile-time rational constant (which the
			// type checker folds to a RationalNumberType, e.g. the whole
			// `-59` in `int256 constant FOO = -59;`). The old `u256_sub(0,
			// x)` lowering treated every site as full-word unsigned
			// subtraction -- spuriously reverting the model for any
			// positive x (0 - x underflows unsigned) even though the real
			// EVM operation (signed two's-complement negation) never
			// reverts except at INT_MIN. Route constant folds to a
			// canonical `i256` literal and everything else to the
			// declared-width `i256_neg` primitive; anything reaching
			// neither branch is not a real >=0.8 program shape, so fail
			// closed instead of re-deriving the old wrong lowering.
			if (unary->annotation().type && unary->annotation().type->category() == Type::Category::RationalNumber)
			{
				auto const* rationalType = dynamic_cast<RationalNumberType const*>(unary->annotation().type);
				if (rationalType && !rationalType->isFractional())
				{
					bigint const value = rationalType->value().numerator();
					bigint const minS256 = -(bigint(1) << 255);
					bigint const maxS256Exclusive = bigint(1) << 255;
					if (value < minS256 || value >= maxS256Exclusive)
						throw UnsupportedSolCore(
							"Constant-folded unary minus produced a value "
							"outside the int256 range.");
					result["kind"] = "i256";
					result["value"] = s2u(s256(value)).str();
					return result;
				}
			}
			if (auto bits = signedOperationBits(unary->annotation().type))
			{
				result["kind"] = "i256_neg";
				result["bits"] = static_cast<int>(*bits);
				markUncheckedContext(result);
				result["operand"] = exportExpr(unary->subExpression());
				return result;
			}
			throw UnsupportedSolCore(
				"Unary minus on a non-constant, non-signed-integer operand "
				"has no faithful SolCore lowering.");
		}
		default:
			throw UnsupportedSolCore("Unsupported unary operator in SolCore exporter.");
		}
	}

	if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
	{
		bool const guardedEffect
			= evalOrderRelevant(conditional->trueExpression()) || evalOrderRelevant(conditional->falseExpression());
		if (!guardedEffect)
		{
			Json result = Json::object();
			result["kind"] = "conditional";
			result["result_type"] = exportResolvedType(conditional->annotation().type);
			result["cond"] = exportExpr(conditional->condition());
			result["true_value"]
				= exportExprCoercedToFixedBytes(conditional->trueExpression(), conditional->annotation().type);
			result["false_value"]
				= exportExprCoercedToFixedBytes(conditional->falseExpression(), conditional->annotation().type);
			return result;
		}

		if (!activeHoistScope || !activeHoistScope->allowed)
			throw UnsupportedSolCore("ternary-arm effects require a once-evaluated statement context.");
		HoistScope* parent = activeHoistScope;
		auto memoIt = parent->memo.find(static_cast<int64_t>(conditional->id()));
		if (memoIt != parent->memo.end())
			return localExpr(memoIt->second);
		Json cond = pinExpressionOnce(conditional->condition(), "conditional_guard");
		std::string resultName
			= "__solcore_evalorder_conditional_result_" + std::to_string(stableSyntheticNodeId(*conditional));
		Json resultType = exportResolvedType(conditional->annotation().type);
		Json initial = Json::object();
		initial["kind"] = "typed_default";
		initial["type"] = resultType;
		Json resultLet = Json::object();
		resultLet["kind"] = "let";
		resultLet["sourceDeclarationId"] = Json();
		resultLet["name"] = resultName;
		resultLet["type"] = resultType;
		resultLet["value"] = std::move(initial);
		parent->statements.emplace_back(std::move(resultLet));

		auto exportArm = [&](Expression const& arm) -> Json
		{
			Json value;
			std::vector<Json> statements;
			{
				HoistScopeGuard armScope({&arm});
				value = exportExprCoercedToFixedBytes(arm, conditional->annotation().type);
				statements = armScope.takeStatements();
			}
			statements.emplace_back(localAssignment(resultName, std::move(value)));
			return blockFromStatements(std::move(statements));
		};

		Json branch = Json::object();
		branch["kind"] = "if";
		branch["cond"] = std::move(cond);
		branch["then"] = exportArm(conditional->trueExpression());
		branch["else"] = exportArm(conditional->falseExpression());
		parent->statements.emplace_back(std::move(branch));
		parent->memo.emplace(static_cast<int64_t>(conditional->id()), resultName);
		return localExpr(resultName);
	}

	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expr))
	{
		bool observable = false;
		for (auto const& component: tuple->components())
			if (component)
				observable = observable || evalOrderRelevant(*component);
		auto exportComponent = [&](Expression const& component, char const* label, size_t index)
		{
			return observable
					   ? captureMeasuredOrderChild(component, std::string(label) + "_" + std::to_string(index))
					   : exportExpr(component);
		};

		// Inline array literals use TupleExpression syntax too. Only the
		// compiler-resolved expected type may distinguish them from genuine
		// tuples. The compiler visits their components left-to-right.
		if (tuple->annotation().type && tuple->annotation().type->category() == Type::Category::Array)
		{
			Json result = Json::object();
			result["kind"] = "array_literal";
			result["type"] = exportResolvedType(tuple->annotation().type);
			result["elements"] = Json::array();
			for (size_t i = 0; i < tuple->components().size(); ++i)
			{
				auto const& component = tuple->components()[i];
				if (!component)
					throw UnsupportedSolCore("Inline array literal contains a missing component.");
				result["elements"].emplace_back(exportComponent(*component, "array_component", i));
			}
			return result;
		}

		// Parenthesized single expressions unwrap; products preserve the
		// compiler's left-to-right component order in explicit capture lets.
		if (tuple->components().size() == 1 && tuple->components().front())
			return exportExpr(*tuple->components().front());
		Json result = Json::object();
		result["kind"] = "tuple";
		result["elements"] = Json::array();
		for (size_t i = 0; i < tuple->components().size(); ++i)
		{
			auto const& component = tuple->components()[i];
			if (component)
				result["elements"].emplace_back(exportComponent(*component, "tuple_component", i));
			else
				result["elements"].emplace_back(Json());
		}
		return result;
	}


	if (auto const* assignment = dynamic_cast<Assignment const*>(&_expr))
		// Assignment used as a VALUE inside a larger expression (e.g.
		// `while ((y = x) > 1)` or `f(x = y)`). The old `assign_expr`
		// lowering was the same bug class as nested `x++`: the OCaml
		// frontend's expression translation returns the value and drops
		// the write (and a state-variable identifier LHS was even
		// misdirected to a LOCAL-name assignment node). Hoist the real
		// assignment statement instead and substitute the assigned value.
		return exportHoistedAssignExpr(*assignment);

	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
	{
		// A view/pure static call that returns one exact storage path is not a
		// first-class reference value: substitute its rooted read immediately.
		// Dynamic indices/keys are captured by the statement scope once, and
		// unresolved/conditional roots continue through the ordinary call
		// lowering (writes through them are refused by the owner guard).
		if (!suppressStorageRefCallResolution)
			if (auto rootedUse = exportResolvedStorageRefUse(*call))
				return std::move(*rootedUse);
		// `new C{value: v, salt: s}(args...)` is a structural deployment
		// node, not an array constructor. The producer binds the exact target
		// contract, effective constructor signature/payability, and creation /
		// runtime artifact identities. Runtime participants are captured once
		// in solc's written order (options, then arguments); the NewExpression
		// itself is a type-level callee and has no runtime evaluation.
		//
		// Non-contract `new` remains an allowlist for dynamically-sized arrays,
		// so no future allocation shape can silently acquire deployment or
		// array semantics.
		{
			Expression const* newCallee = &call->expression();
			auto const* callOptions = dynamic_cast<FunctionCallOptions const*>(newCallee);
			if (callOptions)
				newCallee = &callOptions->expression();
			if (auto const* newExpr = dynamic_cast<NewExpression const*>(newCallee))
			{
				Type const* allocatedType = newExpr->typeName().annotation().type;
				if (allocatedType && allocatedType->category() == Type::Category::Contract)
				{
					auto const* contractType = dynamic_cast<ContractType const*>(allocatedType);
					if (!contractType)
						throw UnsupportedSolCore("Contract deployment has no resolved ContractType.");
					ContractDefinition const& targetContract = contractType->contractDefinition();
					std::string const targetContractId = exportedContractId(targetContract);
					FunctionDefinition const* constructor = targetContract.constructor();
					size_t const constructorArity = constructor ? constructor->parameters().size() : 0;
					Json constructorDisposition = exportConstructorDisposition(targetContract);
					Json constructorArgAbi
						= constructor ? exportConstructorParamsAbi(*constructor) : Json::array();
					if (call->arguments().size() != constructorArity)
						throw UnsupportedSolCore(
							"Contract deployment argument count disagrees with the "
							"resolved target constructor.");

					bool observable = false;
					if (callOptions)
						for (auto const& option: callOptions->options())
							observable = observable || evalOrderRelevant(*option);
					for (auto const& argument: call->arguments())
						observable = observable || evalOrderRelevant(*argument);

					std::vector<Json> evaluatedOptions;
					if (callOptions)
						for (size_t i = 0; i < callOptions->options().size(); ++i)
						{
							std::string const optionName = *callOptions->names()[i];
							if (optionName != "value" && optionName != "salt")
								throw UnsupportedSolCore(
									"Contract deployment option `" + optionName
									+ "` is outside the admitted CREATE/CREATE2 structural model.");
							evaluatedOptions.emplace_back(
								observable
									? pinExpressionOnce(*callOptions->options()[i], "deployment_option_" + optionName)
									: exportExpr(*callOptions->options()[i]));
						}
					std::vector<Json> evaluatedArguments;
					for (size_t i = 0; i < call->arguments().size(); ++i)
						evaluatedArguments.emplace_back(
							observable ? pinExpressionOnce(*call->arguments()[i], "deployment_arg_" + std::to_string(i))
									   : exportExpr(*call->arguments()[i]));

					Json result = Json::object();
					result["kind"] = "contract_deployment";
					result["targetContractId"] = targetContractId;
					result["constructorDisposition"] = std::move(constructorDisposition);
					result["constructorArgAbi"] = std::move(constructorArgAbi);
					Json constructorJson = Json::object();
					constructorJson["contractId"] = targetContractId;
					constructorJson["function"] = constructor ? Json("constructor") : Json();
					constructorJson["payable"]
						= constructor && constructor->stateMutability() == StateMutability::Payable;
					constructorJson["argTypes"] = Json::array();
					if (constructor)
						for (auto const& parameter: constructor->parameters())
							constructorJson["argTypes"].emplace_back(exportTypeName(parameter->typeName()));
					result["constructor"] = std::move(constructorJson);

					result["options"] = Json::array();
					if (callOptions)
						for (size_t i = 0; i < evaluatedOptions.size(); ++i)
						{
							Json option = Json::object();
							option["name"] = *callOptions->names()[i];
							option["value"] = std::move(evaluatedOptions[i]);
							result["options"].emplace_back(std::move(option));
						}
					result["args"] = Json::array();
					for (Json& argument: evaluatedArguments)
						result["args"].emplace_back(std::move(argument));

					Json creationCode = Json::object();
					creationCode["kind"] = "contract_code";
					creationCode["which"] = "creation";
					creationCode["contract"] = targetContractId;
					result["creationCode"] = std::move(creationCode);
					Json runtimeCode = Json::object();
					runtimeCode["kind"] = "contract_code";
					runtimeCode["which"] = "runtime";
					runtimeCode["contract"] = targetContractId;
					result["runtimeCode"] = std::move(runtimeCode);
					return result;
				}
				if (!allocatedType || allocatedType->category() != Type::Category::Array)
					throw UnsupportedSolCore(
						std::string(
							"`new` is only modeled for dynamically-sized array allocations "
							"(`new T[](n)`, `new bytes(n)`, `new string(n)`); this `new` "
							"allocates ")
						+ (allocatedType ? allocatedType->toString(true) : std::string("an unresolved type"))
						+ ", which has no SolCore lowering.");
			}
		}

		// Most EVM type conversions are representation-level no-ops. Two
		// exceptions must retain their source-level semantics here:
		//
		// * conversion TO an enum validates the input against that enum's
		//   cardinality and panics with code 0x21 when it is out of range;
		// * a genuine narrowing integer downcast changes the carried word.
		//
		// Enum cardinality belongs to the typed target at this cast site. Emit
		// it as dedicated metadata on a structurally tagged runtime call rather
		// than asking the Aeneas consumer to recover an enum declaration from a
		// name or from the argument payload.
		if (*call->annotation().kind == FunctionCallKind::TypeConversion)
		{
			if (call->arguments().empty())
				throw UnsupportedSolCore("Empty type conversion in SolCore exporter.");
			Expression const& argExpr = *call->arguments().front();
			Json innerJson = exportExpr(argExpr);
			Type const* targetType = call->annotation().type;
			Type const* sourceType = argExpr.annotation().type;
			if (auto const* targetEnum = dynamic_cast<EnumType const*>(targetType))
			{
				size_t const cardinality = targetEnum->numberOfMembers();
				if (cardinality == 0)
					throw UnsupportedSolCore(
						"Enum conversion target has no members (the type checker should "
						"have rejected this).");
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = "solcore_checked_enum_cast";
				result["runtimeKind"] = "checked_enum_cast";
				result["enumCardinality"] = cardinality;
				result["args"] = Json::array();
				result["args"].emplace_back(std::move(innerJson));
				return result;
			}
			if (targetType && sourceType && targetType->category() == Type::Category::Integer
				&& sourceType->category() == Type::Category::Integer)
			{
				auto const* targetInt = dynamic_cast<IntegerType const*>(targetType);
				auto const* sourceInt = dynamic_cast<IntegerType const*>(sourceType);
				if (targetInt && sourceInt)
				{
					// Signedness-aware conversion matrix (SolCore design:
					// "Signed integer comparison, shift, and division
					// primitives" §4.3). Every SolCore intN value is the
					// full 256-bit SIGN-EXTENDED word (see
					// `signedOperationBits`'s doc comment above), so a
					// narrowing conversion TO a signed type needs
					// `signextend`, never `truncate` -- the old
					// width-only rule silently mod-truncated signed
					// downcasts (e.g. `int128(value)`), which is wrong
					// for any negative or >= 2^127 input. Solidity only
					// allows an explicit int/uint conversion when the
					// width matches OR the signedness matches (never
					// both differing at once -- see
					// `IntegerType::isExplicitlyConvertibleTo`), so this
					// matrix is exhaustive: every remaining case (widening,
					// same type, or a 256-bit sign-only conversion) is a
					// pass-through under the invariant.
					bool const narrowing = targetInt->numBits() < sourceInt->numBits();
					bool const sameWidth = targetInt->numBits() == sourceInt->numBits();
					bool const sameSign = targetInt->isSigned() == sourceInt->isSigned();
					bool const target256 = targetInt->numBits() == 256;
					if ((sameSign && narrowing) || (!sameSign && sameWidth && !target256))
					{
						Json result = Json::object();
						if (targetInt->isSigned())
						{
							result["kind"] = "signextend";
							result["bits"] = static_cast<int>(targetInt->numBits());
						}
						else
						{
							result["kind"] = "truncate";
							result["target_bits"] = static_cast<int>(targetInt->numBits());
						}
						result["value"] = innerJson;
						return result;
					}
				}
			}
			// ── Narrow-`bytesN` width convention (SolCoreBase.ml [BytesW]
			// doc): fixed-bytes conversions. The SolCore value of a
			// `bytesN` is RIGHT-aligned (numeric), so:
			//   * bytesM -> bytesN width changes are REAL shifts on the
			//     carrier (EVM: keep/extend the HIGH bytes; carrier:
			//     `<< 8*(N-M)` widening, `>> 8*(M-N)` narrowing). This
			//     includes bytesN -> bytes32 (the erased full word).
			//   * bytesN <-> uintM / address at the SAME byte size are
			//     identity (both right-aligned) — pass-through below, with
			//     a fail-closed guard on the size equality the type
			//     checker is supposed to guarantee.
			//   * string/hex literals convert by RIGHT-padding to N.
			//   * dynamic `bytes` <-> `bytesN` reinterprets between a byte
			//     ARRAY value and a word value — no pass-through can be
			//     right; refuse until a real truncating lowering exists.
			{
				auto const* targetFB = dynamic_cast<FixedBytesType const*>(targetType);
				auto const* sourceFB = dynamic_cast<FixedBytesType const*>(sourceType);
				if (targetFB && sourceFB && targetFB->numBytes() != sourceFB->numBytes())
				{
					bool const widening = targetFB->numBytes() > sourceFB->numBytes();
					unsigned const diffBits = 8u
											  * (widening ? targetFB->numBytes() - sourceFB->numBytes()
														  : sourceFB->numBytes() - targetFB->numBytes());
					Json shiftLit = Json::object();
					shiftLit["kind"] = "u256";
					shiftLit["value"] = std::to_string(diffBits);
					Json result = Json::object();
					// Widening cannot overflow (v < 2^(8M) implies
					// v << diff < 2^(8N)); it is an explicit full-word shift.
					// Narrowing floors the dropped low bytes and uses u256_shr.
					result["kind"] = widening ? "u256_shl" : "u256_shr";
					if (widening)
						result["bits"] = 256;
					result["lhs"] = innerJson;
					result["rhs"] = shiftLit;
					return result;
				}
				if (targetFB && sourceType && sourceType->category() == Type::Category::StringLiteral)
				{
					auto const* litType = dynamic_cast<StringLiteralType const*>(sourceType);
					std::string const& raw = litType->value();
					if (raw.size() > targetFB->numBytes())
						throw UnsupportedSolCore(
							"String literal longer than the bytesN conversion target "
							"(the type checker should have rejected this).");
					u256 val = 0;
					for (char c: raw)
						val = val * 256 + static_cast<unsigned char>(c);
					for (size_t i = raw.size(); i < targetFB->numBytes(); ++i)
						val *= 256; // right-pad to the declared width
					Json result = Json::object();
					result["kind"] = "u256";
					result["value"] = val.str();
					return result;
				}
				if (targetFB && sourceType && sourceType->category() == Type::Category::Integer)
				{
					auto const* sourceInt = dynamic_cast<IntegerType const*>(sourceType);
					if (sourceInt && sourceInt->numBits() != 8u * targetFB->numBytes())
						throw UnsupportedSolCore(
							"uintM -> bytesN conversion with M != 8N has no "
							"size-preserving lowering (the type checker should "
							"have rejected this).");
				}
				if (sourceFB && targetType && targetType->category() == Type::Category::Integer)
				{
					auto const* targetInt = dynamic_cast<IntegerType const*>(targetType);
					if (targetInt && targetInt->numBits() != 8u * sourceFB->numBytes())
						throw UnsupportedSolCore(
							"bytesN -> uintM conversion with M != 8N has no "
							"size-preserving lowering (the type checker should "
							"have rejected this).");
				}
				// `bytesN(<dynamic bytes / bytes slice>)` (Solidity 0.8.5+):
				// the FIRST N bytes of the array as the bytesN value,
				// zero-padded on the right when shorter. The operand is a
				// byte-ARRAY value, so a pass-through would reinterpret an
				// array as the word — lower through the dedicated typed
				// helper instead. Calldata slices arrive as ArraySliceType
				// (NOT Category::Array), which the old fail-closed guard
				// missed: the conversion silently passed the array value
				// through, losing the reinterpretation entirely.
				if (targetFB && sourceType)
				{
					ArrayType const* sourceArray = nullptr;
					if (auto const* arr = dynamic_cast<ArrayType const*>(sourceType))
						sourceArray = arr;
					else if (auto const* slice = dynamic_cast<ArraySliceType const*>(sourceType))
						sourceArray = &slice->arrayType();
					if (sourceArray)
					{
						if (!sourceArray->isByteArrayOrString())
							throw UnsupportedSolCore(
								"`bytes" + std::to_string(targetFB->numBytes())
								+ "(<non-byte array>)` conversion has no lowering "
								  "(the type checker should have rejected this).");
						Json result = Json::object();
						result["kind"] = "internal_call";
						result["function"]
							= "byte_array_to_bytesn__" + std::to_string(targetFB->numBytes());
						result["args"] = Json::array();
						result["args"].emplace_back(innerJson);
						return result;
					}
				}
				if (sourceFB && targetType
					&& (targetType->category() == Type::Category::Array
						|| targetType->category() == Type::Category::ArraySlice))
					throw UnsupportedSolCore(
						"`bytes(<bytesN>)` conversion to dynamic bytes is not modeled "
						"(narrow-bytesN campaign residual): the result is a byte-array "
						"value the word carrier cannot pass through. Refusing fail-closed.");
			}
			return innerJson;
		}

		// Struct constructor calls: S(field1, field2, ...)
		if (*call->annotation().kind == FunctionCallKind::StructConstructorCall)
		{
			Json result = Json::object();
			result["kind"] = "struct_constructor";
			// Name the constructed record from the compiler-resolved struct
			// type (never the callee's surface spelling): the exported record
			// name is disambiguated when bare struct names collide.
			if (auto const* structType = dynamic_cast<StructType const*>(call->annotation().type))
				result["name"] = exportedStructName(structType->structDefinition());
			else if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
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

		// Dynamic array allocation: retain its exact result type and length,
		// including the typed empty value of `new bytes(0)`.
		if (dynamic_cast<NewExpression const*>(&call->expression()))
		{
			if (!call->annotation().type || call->annotation().type->category() != Type::Category::Array)
				throw UnsupportedSolCore("Non-array new-expression is unsupported here.");
			if (call->arguments().size() != 1)
				throw UnsupportedSolCore("Dynamic array construction requires exactly one length argument.");
			Json result = Json::object();
			result["kind"] = "array_constructor";
			result["type"] = exportResolvedType(call->annotation().type);
			result["length"] = exportExpr(*call->arguments().front());
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

			// sha256(data), keccak256(data), ripemd160(data),
			// ecrecover(h,v,r,s), gasleft(). ripemd160 returns `bytes20`:
			// the generator catalogues it as SC.BytesW 20 and the runtime
			// kernel returns the digest right-aligned — exactly the
			// narrow-bytesN value convention, with ABI/topic left-alignment
			// applied by the BytesW-aware codec arms downstream.
			if (callee->name() == "sha256" || callee->name() == "keccak256" || callee->name() == "ripemd160"
				|| callee->name() == "ecrecover" || callee->name() == "gasleft" || callee->name() == "blockhash")
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
			// virtualCallTargetName). Delegates to lowerInternalCalleeAndArgs
			// for internal-function-typed-parameter specialization (bounded
			// defunctionalization); byte-for-byte identical output to before
			// when the callee has no such parameters.
			auto const* funcDef = dynamic_cast<FunctionDefinition const*>(callee->annotation().referencedDeclaration);
			if (funcDef)
			{
				std::string plainName = virtualCallTargetName(*funcDef);
				FunctionDefinition const& resolvedImpl = resolveInternalCallImplementation(*funcDef);
				auto [calleeName, callArgs] = lowerInternalCalleeAndArgs(*call, resolvedImpl, plainName);
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = std::move(calleeName);
				result["args"] = std::move(callArgs);
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
				std::string plainName = flattenedStaticBaseCallName(*resolved);
				auto [calleeName, callArgs] = lowerInternalCalleeAndArgs(*call, *resolved->target, plainName);
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = std::move(calleeName);
				addStaticBaseCallTarget(result, *resolved);
				result["args"] = std::move(callArgs);
				return result;
			}

			// OpenZeppelin SafeCast narrowing downcasts (`x.toUint128()` via
			// `using SafeCast for uint256`, or the qualified
			// `SafeCast.toUint128(x)` form) — see
			// isKnownOzSafeCastNarrowingDowncast's docstring above. Must run
			// before the using-for/qualified-call `internal_call` branches
			// further below, which would otherwise catch this same call
			// shape first and export it as an opaque call.
			if (auto const* safeCastFuncDef
				= dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
			{
				int targetBits = 0;
				if (isKnownOzSafeCastNarrowingDowncast(*safeCastFuncDef, targetBits))
				{
					Type const* receiverType = memberAccess->expression().annotation().type;
					bool isUsingForCall = receiverType && receiverType->category() != Type::Category::TypeType
										  && receiverType->category() != Type::Category::Module;
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
								throw UnsupportedSolCore(
									"abi.decode: expected a parenthesized type list as the second argument.");

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
									throw UnsupportedSolCore(
										"abi.decode: type-list entry is not a resolvable type name.");
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
							// Producer-authored ABI descriptors for the SAME type
							// list, so struct decodes need no downstream re-parse
							// of type-name strings. Emitted per entry through the
							// exact descriptor exporter calls/events use; a type
							// the descriptor exporter cannot express omits the
							// field, and the frontend fails closed on absence with
							// the type named (exactly as before this field).
							try
							{
								Json decodeAbi = Json::array();
								for (auto const& component: typesTuple->components())
								{
									auto const* typeType
										= dynamic_cast<TypeType const*>(component->annotation().type);
									decodeAbi.emplace_back(exportAbiDescriptor(
										"", typeType->actualType(), false));
								}
								result["decode_abi"] = std::move(decodeAbi);
							}
							catch (UnsupportedSolCore const&)
							{
								// fall through without decode_abi
							}
							return result;
						}

						Json result = Json::object();
						result["kind"] = "internal_call";
						result["function"] = "abi_" + memberAccess->memberName();
						result["args"] = Json::array();
						// abi.encodeCall's FIRST argument names the callee. When
						// it is a statically resolved DECLARATION (the namespace
						// forms `I.echo` / `C.echo` / `L.increment`) there is no
						// receiver address to export, and encodeCall never needs
						// one — carry the selector constant instead of failing
						// the function closed on a bare contract identifier.
						// Bound members and function-typed VALUES are untouched
						// (exportDeclaredCalleeSelector returns nullopt for
						// Kind::External) and keep their extfn_pack lowering.
						if (memberAccess->memberName() == "encodePacked")
						{
							result["argTypes"] = Json::array();
							for (auto const& arg: call->arguments())
								result["argTypes"].emplace_back(
									exportByteHelperArgumentType(arg->annotation().type));
						}
						size_t abiArgIndex = 0;
						bool declaredCalleeSelectorEmitted = false;
						if (memberAccess->memberName() == "encodeCall" && !call->arguments().empty())
							if (auto selectorNode = exportDeclaredCalleeSelector(*call->arguments().front()))
							{
								result["args"].emplace_back(std::move(*selectorNode));
								declaredCalleeSelectorEmitted = true;
							}
						bool observable = false;
						for (size_t i = 0; i < call->arguments().size(); ++i)
							if (!(declaredCalleeSelectorEmitted && i == 0))
								observable = observable || evalOrderRelevant(*call->arguments()[i]);

						for (auto const& arg: call->arguments())
						{
							// No catch here: an argument this pipeline can't lower must
							// propagate (to exportBody's catch-all, which turns it into a
							// diagnosable "unsupported_body" marker), not silently become
							// a bare `0` with no trace at all. abi.decode itself never
							// reaches this loop (handled above, out-of-band); this is only
							// abi.encode/abi.encodePacked/abi.encodeWithSignature now.
							if (!(declaredCalleeSelectorEmitted && abiArgIndex == 0))
							{
								Json argValue
									= observable
										  ? captureMeasuredOrderChild(
												*arg, "abi_arg_" + std::to_string(abiArgIndex))
										  : exportExpr(*arg);

								// abi.encodePacked contributes each argument's
								// RAW bytes: a narrow bytesN argument contributes
								// exactly its N bytes (its left-aligned register's
								// top N bytes), NOT a 32-byte word. The generic
								// packed helper encodes by the erased Lean type
								// (32 bytes per word value), so wrap narrow
								// fixed-bytes arguments in the width-mangled
								// byte-truncation helper the generator emits
								// (narrow-bytesN width convention). abi.encode*'s
								// other forms either monomorphize (static-word
								// arm, BytesW-aware) or stay opaque (sound).
								Type const* argType = arg->annotation().type;
								auto const* argFB = memberAccess->memberName() == "encodePacked"
														? dynamic_cast<FixedBytesType const*>(argType)
														: nullptr;
								if (argFB && argFB->numBytes() < 32)
								{
									Json wrapped = Json::object();
									wrapped["kind"] = "internal_call";
									wrapped["function"] = "bytesn_packed__" + std::to_string(argFB->numBytes());
									wrapped["args"] = Json::array();
									wrapped["args"].emplace_back(std::move(argValue));
									result["args"].emplace_back(std::move(wrapped));
								}
								else
									result["args"].emplace_back(std::move(argValue));
							}
							++abiArgIndex;
						}
						return result;
					}
				}
			}

			// Array pop: arr.pop() — BUILTIN pop only, gated on the callee's
			// FunctionType::Kind (§2c case 1): a using-for `pop` (e.g. a
			// library function attached to a memory array, where no builtin
			// pop exists) resolves to a FunctionDefinition and must fall
			// through to the using-for internal-call branch below instead of
			// being intercepted BY NAME.
			if (memberAccess->memberName() == "pop")
			{
				auto const* popFnType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type);
				Type const* baseType = memberAccess->expression().annotation().type;
				if (popFnType && popFnType->kind() == FunctionType::Kind::ArrayPop && baseType
					&& baseType->category() == Type::Category::Array)
				{
					Json result = Json::object();
					result["kind"] = "array_pop";
					result["base"] = exportExpr(memberAccess->expression());
					return result;
				}
			}

			// Array push in EXPRESSION position: `arr.push()` returning a
			// storage reference to the new element (`xs.push().a = 7`), or a
			// push whose result is otherwise consumed. §2c case 2: the old
			// lowering fabricated a default `"value":"0"` array_push node
			// here — shape-invalid and value-fabricating, caught only by the
			// generator's incidental catalog gate. Fail closed loudly instead;
			// statement-position push is fully handled in exportStmt. Gated on
			// FunctionType::Kind like `pop` above so a using-for `push` falls
			// through to the using-for internal-call branch below.
			if (memberAccess->memberName() == "push")
			{
				auto const* pushFnType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type);
				if (pushFnType && pushFnType->kind() == FunctionType::Kind::ArrayPush)
					throw UnsupportedSolCore(
						"array push in expression position (push used as a value, e.g. "
						"`arr.push().field = v`) is not modeled; refusing to fabricate a "
						"default-valued array_push node.");
			}

			// Attached (`using for`) calls are identified by solc's bound-first-
			// argument FunctionType fact. This covers internal, public and
			// external library functions and emits the receiver exactly once.
			if (auto usingForCall = exportUsingForCall(*call, *memberAccess))
				return *usingForCall;

			// External contract call: contract.method(args)
			Type const* baseType = memberAccess->expression().annotation().type;
			if (baseType && baseType->category() == Type::Category::Contract)
			{
				if (auto externalCall = exportExternalContractCall(*call, *memberAccess, false))
					return *externalCall;
			}

			// Library-qualified or type-qualified function call: L.f(args)
			// The member access resolves to a FunctionDefinition when calling through a library/type namespace
			if (auto const* funcDef
				= dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
			{
				Json result = Json::object();
				result["kind"] = "internal_call";
				std::string plainName = isPublicLibraryStructuralStorageFunction(*funcDef)
											? storageRefInternalEntryName(*funcDef)
											: exportedFunctionName(*funcDef);
				auto [calleeName, callArgs] = lowerInternalCalleeAndArgs(*call, *funcDef, std::move(plainName));
				result["function"] = std::move(calleeName);
				addInternalLibraryCallContractId(result, *funcDef);
				result["args"] = std::move(callArgs);
				return result;
			}
		}

		// Options-aware low-level calls. Staticcall admits only gas; its
		// evaluated numeric value is intentionally absent from the model.
		if (auto const* options = dynamic_cast<FunctionCallOptions const*>(&call->expression()))
		{
			if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&options->expression()))
			{
				if (auto externalCall = exportExternalContractCall(*call, *memberAccess, false))
					return *externalCall;

				if (isOptionsAwareLowLevelCall(memberAccess->memberName()))
				{
					validateLowLevelCallOptions(memberAccess->memberName(), *options);
					CallEvaluation evaluation = exportCallEvaluation(*call, memberAccess->expression(), options);
					Json result = Json::object();
					result["kind"] = "low_level_call";
					result["callKind"] = lowLevelCallKindString(memberAccess->memberName());
					result["target"] = std::move(evaluation.target);

					Json valueExpr = u256Literal("0");
					for (size_t i = 0; i < options->names().size(); ++i)
						if (*options->names()[i] == "value")
							valueExpr = std::move(evaluation.options[i]);
					result["value"] = std::move(valueExpr);

					if (!evaluation.arguments.empty())
						result["data"] = std::move(evaluation.arguments.front());
					else
					{
						result["data"] = emptyBytesLiteral();
					}
					return result;
				}
			}
		}

		// Check for low-level calls without options:
		// address.call(data), address.delegatecall(data), or address.staticcall(data).
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
		{
			if (isOptionsAwareLowLevelCall(memberAccess->memberName()))
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
						result["data"] = emptyBytesLiteral();
					}
					return result;
				}
			}
		}

		// address.send(x) / address.transfer(x)
		// (spechunt-reject-buckets-design §2b). `.send` previously had no
		// branch, fell into the generic member-call fallback below and was
		// exported as a bare `internal_call send` with the RECEIVER SILENTLY
		// DROPPED (caught only by the generator's fail-closed catalog gate).
		// Lower `a.send(x)` to the existing low_level_call node — a strict
		// over-approximation of the 2300-gas-stipend callee (the modeled
		// arbitrary callee can do strictly more) — and project the success
		// boolean via the same tuple_get(·, 0) node the tuple-destructuring
		// lowering already uses for `(bool ok, ) = addr.call(...)`.
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
		{
			auto const* calleeFnType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type);
			if (calleeFnType && calleeFnType->kind() == FunctionType::Kind::Send)
			{
				if (call->arguments().size() != 1)
					throw UnsupportedSolCore("address.send expects exactly one value argument.");
				Json lowLevel = Json::object();
				lowLevel["kind"] = "low_level_call";
				lowLevel["callKind"] = lowLevelCallKindString("call");
				lowLevel["target"] = exportExpr(memberAccess->expression());
				lowLevel["value"] = exportExpr(*call->arguments().front());
				lowLevel["data"] = emptyBytesLiteral();

				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = "tuple_get";
				result["args"] = Json::array();
				result["args"].emplace_back(std::move(lowLevel));
				Json index = Json::object();
				index["kind"] = "u256";
				index["value"] = "0";
				result["args"].emplace_back(std::move(index));
				return result;
			}
			// `.transfer` has revert-on-failure semantics `.send` does not;
			// no corpus case needs it. A loud refusal beats the silent
			// receiver-drop of the generic fallback below.
			if (calleeFnType && calleeFnType->kind() == FunctionType::Kind::Transfer)
				throw UnsupportedSolCore(
					"address.transfer is not modeled (revert-on-failure semantics differ "
					"from address.send); refusing to lower it to a bare internal_call.");
		}
	}

	// Fallback for FunctionCall that doesn't match known patterns
	if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
	{
		// Call through an EXTERNAL function-pointer VALUE: `fn(args)` where
		// `fn` is a parameter, local, storage value, try-binding, or any
		// other expression of external function type that solc did NOT
		// statically resolve to a declaration (statically-resolved member
		// calls `I(t).echo(v)` matched exportExternalContractCall earlier and
		// never reach this fallback). This is a genuine external call whose
		// target address and selector are runtime data; export a dedicated
		// node so the generator routes it through the same adversarial
		// external-call machinery as every other external call. Declared
		// parameter/return types are read off the FunctionType itself —
		// never inferred from the argument expressions.
		{
			auto const* calleeFnType = dynamic_cast<FunctionType const*>(call->expression().annotation().type);
			bool calleeIsStaticDecl = false;
			if (auto const* calleeIdent = dynamic_cast<Identifier const*>(&call->expression()))
				calleeIsStaticDecl
					= dynamic_cast<FunctionDefinition const*>(calleeIdent->annotation().referencedDeclaration)
					  != nullptr;
			else if (auto const* calleeMember = dynamic_cast<MemberAccess const*>(&call->expression()))
				calleeIsStaticDecl
					= dynamic_cast<FunctionDefinition const*>(calleeMember->annotation().referencedDeclaration)
					  != nullptr;
			if (calleeFnType && calleeFnType->kind() == FunctionType::Kind::External && !calleeIsStaticDecl)
			{
				if (dynamic_cast<FunctionCallOptions const*>(&call->expression()))
					throw UnsupportedSolCore(
						"external function-pointer call with call options "
						"({value:...}/{gas:...}) is not modeled.");
				if (calleeFnType->takesArbitraryParameters())
					throw UnsupportedSolCore(
						"external function-pointer call with arbitrary parameters is not modeled.");
				Json result = Json::object();
				result["kind"] = "extfn_call";
				result["callMode"] = externalCallMode(*calleeFnType, false);
				result["fn"] = exportExpr(call->expression());
				result["args"] = Json::array();
				for (auto const& arg: call->arguments())
					result["args"].emplace_back(exportExpr(*arg));
				result["argTypes"] = Json::array();
				for (auto const* paramType: calleeFnType->parameterTypes())
					result["argTypes"].emplace_back(exportResolvedType(paramType));
				result["returnTypes"] = Json::array();
				for (auto const* retType: calleeFnType->returnParameterTypes())
					result["returnTypes"].emplace_back(exportResolvedType(retType));
				result["mutability"] = (calleeFnType->stateMutability() == StateMutability::View
										|| calleeFnType->stateMutability() == StateMutability::Pure)
										   ? "view"
										   : "stateful";
				return result;
			}
		}

		// Indirect internal-function calls dispatch through the producer's
		// finite candidate table. The singleton specialization remains a fast
		// path: while exporting a specialized sibling its erased parameter is
		// still rewritten directly to the one bound target.
		auto const* internalCalleeType = dynamic_cast<FunctionType const*>(call->expression().annotation().type);
		if (internalCalleeType && internalCalleeType->kind() == FunctionType::Kind::Internal)
		{
			if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
				if (auto const* varDecl
					= dynamic_cast<VariableDeclaration const*>(callee->annotation().referencedDeclaration))
				{
					auto bindingIt = activeFnPtrBindings.find(varDecl);
					if (bindingIt != activeFnPtrBindings.end())
					{
						Json result = Json::object();
						result["kind"] = "internal_call";
						result["function"] = bindingIt->second->targetExportedName;
						result["args"] = Json::array();
						for (auto const& arg: call->arguments())
							result["args"].emplace_back(exportExpr(*arg));
						return result;
					}
				}
			if (!call->names().empty())
				throw UnsupportedSolCore(
					"Named arguments on an indirect internal-function call are refused; "
					"the source-order-to-parameter-order permutation is not exported.");
			if (internalCalleeType->takesArbitraryParameters())
				throw UnsupportedSolCore("Indirect internal-function call has an open arbitrary-parameter type.");
			Json wireType = exportResolvedType(internalCalleeType);
			InternalFnTableRecord const& table = internalFnTablesByFingerprint.at(internalCalleeType->richIdentifier());
			Json result = Json::object();
			result["kind"] = "internal_fn_call";
			result["table"] = table.tableId;
			result["dispatcher"] = table.dispatcherName;
			result["type"] = wireType;
			result["fn"] = exportExpr(call->expression());
			result["args"] = Json::array();
			for (auto const& arg: call->arguments())
				result["args"].emplace_back(exportExpr(*arg));
			return result;
		}

		// Generic function call — export as internal_call with best-effort name
		Json result = Json::object();
		result["kind"] = "internal_call";
		if (auto const* callee = dynamic_cast<Identifier const*>(&call->expression()))
			result["function"] = callee->name();
		else if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
		{
			if (auto const* funcDef
				= dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
				result["function"] = exportedFunctionName(*funcDef);
			else
			{
				// Same fail-open class as the identifier-callee guard above,
				// through a MEMBER access instead: a call through an
				// internal-function-typed member (e.g. a function pointer
				// stored in a struct field, `s.f(...)`) used to name-punt to
				// the bare member name — silently mis-binding to any real
				// internal function sharing that name. There is no static
				// target to resolve here (member function pointers are never
				// specialization-bound), so fail closed.
				auto const* memberFnType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type);
				if (memberFnType && memberFnType->kind() == FunctionType::Kind::Internal)
					throw UnsupportedSolCore(
						"indirect call through an internal function value ('" + memberAccess->memberName()
						+ "', a member access) cannot be resolved to a static target; not modeled.");
				// [P0 fail-closed: no callee name-punt for unmodeled bare
				// low-level calls] A BARE low-level call reaching here was not
				// matched by the low-level-call arms above. Punting to the bare
				// member name would export an unresolvable internal callee and
				// model the function as unconditionally reverting.
				//
				// Narrowly typed to the four Bare* kinds precisely so the
				// legitimate member-name path is untouched: a corpus-wide scan
				// of every non-local `internal_call` name shows this branch is
				// load-bearing for real cross-unit callees (`pushPub`,
				// `sumPub`, `isOff`, `publicLength`, `bump`,
				// `withdrawalAllowed`), none of which is a Bare* kind.
				if (memberFnType
					&& (memberFnType->kind() == FunctionType::Kind::BareCall
						|| memberFnType->kind() == FunctionType::Kind::BareCallCode
						|| memberFnType->kind() == FunctionType::Kind::BareDelegateCall
						|| memberFnType->kind() == FunctionType::Kind::BareStaticCall))
					throw UnsupportedSolCore(
						"Low-level `." + memberAccess->memberName()
						+ "(...)` has no SolCore "
						  "lowering (only `call` and `delegatecall` are modeled, as "
						  "`low_level_call`). Refusing instead of exporting the bare member name "
						  "as a callee that resolves to nothing and would be modeled as an "
						  "unconditional revert.");
				result["function"] = memberAccess->memberName();
			}
		}
		else
		{
			// [P0 fail-closed: no `unknown_call` name-punt]
			//
			// This branch used to emit `internal_call unknown_call` — a callee
			// name that resolves to no function in the unit and to no runtime
			// helper. The generator does not reject it; it emits
			//
			//   internal_call_unknown_call … :=
			//     Result.fail (.solcoreRequire
			//       "unresolved internal runtime helper: internal_call_unknown_call")
			//
			// Falling through to an unresolved callee would model the call as
			// unconditionally reverting rather than refusing it. Every
			// FunctionCallOptions shape must therefore be consumed by a
			// dedicated options-aware lowering or fail closed here.
			if (dynamic_cast<FunctionCallOptions const*>(&call->expression()))
				throw UnsupportedSolCore(
					"Call with call-options (`{value: …}` / `{gas: …}` / `{salt: …}`) on a "
					"callee this exporter has no options-aware lowering for is not modeled "
					"(`call`, `delegatecall`, and gas-only `staticcall` are the supported "
					"low-level shapes). Refusing instead of exporting an unresolvable callee "
					"name that would be modeled as an unconditional revert.");
			throw UnsupportedSolCore(
				"Call through a callee expression that is neither a plain identifier nor a "
				"member access (e.g. a function value selected by a ternary, or a "
				"parenthesised function expression) cannot be resolved to a static target; not "
				"modeled. Refusing instead of exporting an unresolvable callee name that would "
				"be modeled as an unconditional revert.");
		}
		if (
			internalCalleeType
			&& (
				internalCalleeType->kind() == FunctionType::Kind::BytesConcat
				|| internalCalleeType->kind() == FunctionType::Kind::StringConcat
			)
		)
		{
			result["argTypes"] = Json::array();
			for (auto const& arg: call->arguments())
				result["argTypes"].emplace_back(
					exportByteHelperArgumentType(arg->annotation().type));
		}
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
	throw UnsupportedSolCore("exportExpr: unrecognized expression node (no matching Expression subtype)");
}

Json exportRevertPayload(
	FunctionCall const& _call,
	bool _unconditionalPayload,
	bool _includeRuntimeArgs)
{
	// `_errorCall` is the CALL WHOSE ARGUMENTS BELONG TO THE ERROR: the outer
	// `_call` itself when its callee IS the error (`revert Err(x)` routed
	// here, error-callee `require`-forms), but the INNER reason-position call
	// for `require(cond, Err(x))` / `revert(Err(x))`. The old single-capture
	// version always read `_call.arguments()`, so a reason-position custom
	// error exported `args = [<condition>, <raw error call>]` — a fabricated
	// payload (and the raw-error-call export is what made base-qualified
	// `require(c, Base.Err(x))` throw spuriously).
	auto exportCustomError = [&](ErrorDefinition const& _error, FunctionCall const& _errorCall)
	{
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
		// Statement payloads retain runtime arguments. Dispatch metadata only
		// needs the custom-error schema used to decode returndata, and running
		// expression lowering from that contract-level prepass would bypass
		// the function body's hoist scope.
		bool pinLegacyOrder = _includeRuntimeArgs && _unconditionalPayload;
		if (pinLegacyOrder)
		{
			pinLegacyOrder = false;
			for (auto const& argument: _errorCall.arguments())
				pinLegacyOrder = pinLegacyOrder || evalOrderRelevant(*argument);
		}
		if (pinLegacyOrder)
		{
			if (!activeCompilerStack)
				throw UnsupportedSolCore(
					"custom-error arguments with observable evaluation order "
					"require an explicit legacy codegen identity.");
			if (activeCompilerStack->viaIR())
				throw UnsupportedSolCore(
					"custom-error argument order is not admitted under via-IR; "
					"only the measured legacy left-to-right row is supported.");
		}

		if (_includeRuntimeArgs)
		{
			Json args = Json::array();
			for (size_t i = 0; i < _errorCall.arguments().size(); ++i)
				args.emplace_back(
					pinLegacyOrder
						? captureMeasuredOrderChild(
							  *_errorCall.arguments()[i], "custom_error_arg_" + std::to_string(i))
						: exportExpr(*_errorCall.arguments()[i]));
			if (!args.empty())
				result["error"]["args"] = std::move(args);
		}
		if (_errorCall.arguments().size() == 1)
		{
			if (auto const* message = dynamic_cast<Literal const*>(_errorCall.arguments().front().get()))
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
		return exportCustomError(*errorDef, _call);

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
	auto exportReasonArg = [&](Expression const& _reasonArg) -> Json
	{
		if (auto const* errorCall = dynamic_cast<FunctionCall const*>(&_reasonArg))
		{
			if (auto const* errorDef = dynamic_cast<ErrorDefinition const*>(
					dynamic_cast<Identifier const*>(&errorCall->expression())
						? dynamic_cast<Identifier const*>(&errorCall->expression())->annotation().referencedDeclaration
					: dynamic_cast<MemberAccess const*>(&errorCall->expression())
						? dynamic_cast<MemberAccess const*>(&errorCall->expression())
							  ->annotation()
							  .referencedDeclaration
						: nullptr))
				return exportCustomError(*errorDef, *errorCall);
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

ModifierDefinition const*
resolveModifierDefinition(FunctionDefinition const& _function, ModifierInvocation const& _modifierInvocation)
{
	auto modifierDefinition
		= dynamic_cast<ModifierDefinition const*>(_modifierInvocation.name().annotation().referencedDeclaration);
	if (!modifierDefinition)
		return nullptr;

	if (_function.isFree())
		return modifierDefinition;

	ContractDefinition const* contract = _function.annotation().contract;
	if (!contract)
		return modifierDefinition;

	if (_modifierInvocation.name().annotation().requiredLookup.set()
		&& *_modifierInvocation.name().annotation().requiredLookup == VirtualLookup::Virtual)
		return &modifierDefinition->resolveVirtual(*contract);

	return modifierDefinition;
}

Json modifierParameterBindings(
	ModifierDefinition const& _modifierDefinition, std::vector<ASTPointer<Expression>> const* _arguments)
{
	Json bindings = Json::array();
	auto const& params = _modifierDefinition.parameters();
	size_t argCount = _arguments ? _arguments->size() : 0;
	for (size_t i = 0; i < params.size(); ++i)
	{
		VariableDeclaration const& param = *params[i];
		Json letStmt = Json::object();
		letStmt["kind"] = "let";
		letStmt["sourceDeclarationId"] = std::to_string(param.id());
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

/// The combined value written by an assignment: `_rhsJson` for `=`, or
/// `<current> op <rhs>` (at the assigned expression's own type) for the
/// compound operators. Shared by statement-level exportAssignment and the
/// nested-assignment hoisting path (exportHoistedAssignExpr).
Json compoundAssignmentValue(Token _op, Type const* _lhsType, Json const& _current, Json const& _rhsJson)
{
	Json value = Json::object();
	// Compound assignment operates at the assigned expression's own type
	// (`a += b` is `a = a + b` at type(a)); the type checker guarantees `b`
	// is implicitly convertible to it. This was previously the ONLY
	// signedness check missing from the whole signed-arithmetic family
	// (SolCore design §3-2): every signed `+=`/`-=`/`*=`/`/=`/`%=`/`>>=`
	// exported as an unsigned op with no width tag at all -- fail-open, not
	// fail-closed. `signedOperationBits` closes that gap the same way it
	// closes it for the `BinaryOperation` switch above.
	auto bits = signedOperationBits(_lhsType);
	switch (_op)
	{
	case Token::Assign:
		return _rhsJson;
	case Token::AssignAdd:
		if (bits)
		{
			value["kind"] = "i256_add";
			value["bits"] = static_cast<int>(*bits);
			markUncheckedContext(value);
		}
		else
		{
			value["kind"] = "u256_add";
			markUncheckedContext(value);
			tagUnsignedArithWidth(value, _lhsType);
		}
		break;
	case Token::AssignSub:
		if (bits)
		{
			value["kind"] = "i256_sub";
			value["bits"] = static_cast<int>(*bits);
			markUncheckedContext(value);
		}
		else
		{
			value["kind"] = "u256_sub";
			markUncheckedContext(value);
			tagUnsignedArithWidth(value, _lhsType);
		}
		break;
	case Token::AssignMul:
		if (bits)
		{
			value["kind"] = "i256_mul";
			value["bits"] = static_cast<int>(*bits);
			markUncheckedContext(value);
		}
		else
		{
			value["kind"] = "u256_mul";
			markUncheckedContext(value);
			tagUnsignedArithWidth(value, _lhsType);
		}
		break;
	case Token::AssignDiv:
		if (bits)
		{
			value["kind"] = "i256_div";
			value["bits"] = static_cast<int>(*bits);
			markUncheckedContext(value);
		}
		else
			value["kind"] = "u256_div";
		break;
	case Token::AssignMod:
		if (bits)
		{
			value["kind"] = "i256_mod";
			value["bits"] = static_cast<int>(*bits);
		}
		else
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
		if (bits)
		{
			if (*bits < 256)
				throw UnsupportedSolCore("Narrow signed '<<=' has no faithful SolCore lowering yet.");
			value["kind"] = "u256_shl";
			value["bits"] = 256;
		}
		else
		{
			value["kind"] = "u256_shl";
			tagUnsignedArithWidth(value, _lhsType);
		}
		break;
	case Token::AssignSar:
		value["kind"] = bits ? "i256_sar" : "u256_shr";
		break;
	default:
		throw UnsupportedSolCore("Unsupported assignment operator in SolCore exporter.");
	}
	value["lhs"] = _current;
	value["rhs"] = _rhsJson;
	return value;
}

Json exportAssignment(Expression const& _lhs, Token _op, Expression const& _rhs)
{
	auto mkDirectStorage = [&](std::string const& _field, Json const& _value)
	{
		Json result = Json::object();
		result["kind"] = "storage_set";
		result["field"] = _field;
		result["value"] = _value;
		return result;
	};

	auto mkDirectMap = [&](std::vector<std::string> const& _path, Json const& _key, Json const& _value)
	{
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

	auto compoundValue = [&](Json const& _current, Json const& _rhsJson)
	{ return compoundAssignmentValue(_op, _lhs.annotation().type, _current, _rhsJson); };

	if (auto const* identifier = dynamic_cast<Identifier const*>(&_lhs))
	{
		auto const* declaration
			= dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (declaration && storageRefValueLocals.count(declaration))
		{
			if (isStorageRefParameter(declaration))
				throw UnsupportedSolCore(
					"Whole-root assignment cannot reseat a storage-reference parameter.");
			if (_op != Token::Assign)
				throw UnsupportedSolCore("Compound assignment cannot rebind a storage-reference value.");
			auto reference = exportStorageRefValue(_rhs);
			if (!reference)
				throw UnsupportedSolCore("Storage-reference rebind did not resolve to an exact typed path.");
			Json result = Json::object();
			result["kind"] = "assign";
			result["name"] = declaration->name().empty() ? identifier->name() : declaration->name();
			result["value"] = std::move(*reference);
			return result;
		}
	}

	// A zero-argument storage-array push used as an assignment lvalue is a
	// producer-known reference operation, not a call to a Solidity function
	// named `generic_assign`. Preserve that identity from solc's FunctionType
	// annotation so the frontend receives the exact typed functional push
	// helper and can re-root nested field/array/mapping bases.
	MemberAccess const* typedPushLvalue = nullptr;
	if (_op == Token::Assign)
		if (auto const* call = dynamic_cast<FunctionCall const*>(&_lhs); call && call->arguments().empty())
			if (auto const* member = dynamic_cast<MemberAccess const*>(&call->expression()))
			{
				auto const* functionType = dynamic_cast<FunctionType const*>(member->annotation().type);
				auto const* arrayType = dynamic_cast<ArrayType const*>(member->expression().annotation().type);
				if (member->memberName() == "push" && functionType
					&& functionType->kind() == FunctionType::Kind::ArrayPush && arrayType
					&& arrayType->location() == DataLocation::Storage && arrayType->isDynamicallySized())
					typedPushLvalue = member;
			}

	// A member/index assignment rooted at `arr.push()` must retain the
	// returned slot as a first-class storage reference. Direct
	// `arr.push() = value` keeps the cheaper typed-push lowering below.
	FunctionCall const* pushReturnedStorageLvalue
		= typedPushLvalue ? nullptr : storageArrayPushLValueRoot(_lhs);

	auto const* lhsTuple = dynamic_cast<TupleExpression const*>(&_lhs);
	bool const structuralTupleAssignment
		= _op == Token::Assign && lhsTuple && !lhsTuple->isInlineArray();
	bool const ordered = evalOrderRelevant(_rhs)
						 || evalOrderRelevant(typedPushLvalue ? typedPushLvalue->expression() : _lhs);
	Json rhsJson;
	if (ordered && !structuralTupleAssignment)
	{
		// Legacy and via-IR agree here: Assignment evaluates the RHS before
		// visiting its lvalue, and IndexAccess visits base before key. The
		// typed push itself is the write emitted below, so only its base is an
		// lvalue child to pin; evaluating the whole push here would duplicate
		// the mutation.
		rhsJson = captureMeasuredOrderChild(_rhs, "assignment_rhs");
		if (typedPushLvalue)
			pinAssignmentLValueChildren(typedPushLvalue->expression(), "assignment_lhs_push_base");
		else if (!pushReturnedStorageLvalue)
			pinAssignmentLValueChildren(_lhs, "assignment_lhs");
	}
	else
		// tuple_assign must retain its structural RHS/destination trees: the
		// TupleExpression arm pins RHS components left-to-right, while the
		// dedicated tuple-assignment lowering owns destination evaluation and
		// right-to-left stores.
		rhsJson = exportExpr(_rhs);
	bool const residualStorageLvalue = hasResidualStorageRefRoot(_lhs);
	if (pushReturnedStorageLvalue || residualStorageLvalue)
	{
		auto reference = pushReturnedStorageLvalue ? exportOrderedStorageRefValueUse(_lhs)
												  : exportStorageRefValue(_lhs);
		if (!reference)
			throw UnsupportedSolCore(
				pushReturnedStorageLvalue
					? "Storage-array push lvalue did not resolve to its exact returned slot."
					: "Residual storage-reference lvalue did not resolve to an exact typed path.");
		std::string tempName = "__solcore_residual_sref_" + std::to_string(stableSyntheticNodeId(_lhs));
		Json letReference = Json::object();
		letReference["kind"] = "let";
		letReference["sourceDeclarationId"] = Json();
		letReference["name"] = tempName;
		letReference["type"] = storageRefWireType(_lhs.annotation().type);
		letReference["value"] = std::move(*reference);

		Json localReference = localExpr(tempName);
		Json set = Json::object();
		set["kind"] = "storage_ref_set";
		set["referentType"] = exportResolvedType(_lhs.annotation().type, true);
		set["reference"] = localReference;
		set["value"] = compoundValue(storageRefGetJson(std::move(localReference), _lhs.annotation().type), rhsJson);

		Json block = Json::object();
		block["kind"] = "block";
		block["statements"] = Json::array();
		block["statements"].emplace_back(std::move(letReference));
		block["statements"].emplace_back(std::move(set));
		return block;
	}
	if (typedPushLvalue)
	{
		auto rootedBase = exportResolvedStorageRefUse(typedPushLvalue->expression());
		if (!rootedBase)
			throw UnsupportedSolCore(
				"Storage-array push assignment base did not resolve to an exact rooted storage path.");

		// `arr.push() = value` appends a default element and immediately
		// overwrites that returned reference. With no intervening observation,
		// this is exactly the checked typed push of `value`; representing it as
		// array_push_expr preserves nested writeback without leaving the
		// producer-only generic_assign pseudo-helper for catalog resolution.
		Json result = Json::object();
		result["kind"] = "expr";
		Json push = Json::object();
		push["kind"] = "internal_call";
		push["function"] = "array_push_expr";
		push["args"] = Json::array();
		push["args"].emplace_back(std::move(*rootedBase));
		push["args"].emplace_back(std::move(rhsJson));
		result["value"] = std::move(push);
		return result;
	}



	if (auto const* identifier = dynamic_cast<Identifier const*>(&_lhs))
	{
		auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (!decl)
			throw UnsupportedSolCore("Assignment target without declaration.");
		// EIP-1153 `transient` state variable write (incl. compound `op=`,
		// whose current-value read must also come from the transient
		// space): a dedicated statement, NEVER storage_set — see the
		// transient_get arm in exportExpr.
		if (isTransientStateVar(decl))
		{
			Json current = Json::object();
			current["kind"] = "transient_get";
			current["field"] = decl->name();
			Json result = Json::object();
			result["kind"] = "transient_set";
			result["field"] = decl->name();
			result["value"] = compoundValue(current, rhsJson);
			return result;
		}
		if (decl->immutable())
		{
			Json value = _op == Token::Assign ? std::move(rhsJson) : compoundValue(immutableGet(*decl), rhsJson);
			return immutableSet(*decl, std::move(value));
		}
		if (decl->isStateVariable())
		{
			Json current = Json::object();
			current["kind"] = "storage_get";
			current["field"] = decl->name();
			return mkDirectStorage(decl->name(), compoundValue(current, rhsJson));
		}

		// [E2(b) — single-assignment storage-pointer binding] A bare
		// statement-position `p = <rhs>;` on a declare-then-assign storage
		// pointer local (exactly one assignment in the whole function — see
		// storageRefSingleAssignBindable) is the local's BINDING initializer:
		// solc's definite-assignment rule dominates every use with it.
		// Resolve the RHS exactly like a declaration initializer and register
		// the alias; the statement itself becomes the key-snapshot block.
		// The composed target goes through the SAME shrink guard as a
		// declaration bind (E0(b)). An unresolvable RHS falls back to
		// today's value-copy assign — value-correct for reads — UNLESS the
		// function writes through the local, in which case this site owns
		// the E0(a') refusal the declaration lowering deferred here.
		if (_op == Token::Assign && !decl->isStateVariable()
			&& decl->referenceLocation() == VariableDeclaration::Location::Storage
			&& storageRefSingleAssignBindable.count(decl) && !storageRefAliasTargets.count(decl))
		{
			if (auto resolved = resolveStorageRefInitializer(_rhs, *decl))
			{
				if (storageRefTargetRacesArrayShrink(resolved->first))
					throw UnsupportedSolCore(
						"Storage reference bound to an element of `" + resolved->first.root
						+ "`, which this function can shrink (`pop()`/`delete`/whole "
						  "reassignment), is not modeled: solc fixes the pointer's SLOT at "
						  "this binding access and raw-sloads it afterwards with no length "
						  "recheck, so a later shrink leaves the reference reading and "
						  "writing the freed slot, while this model's live re-projection "
						  "raises Panic 0x32. Refusing instead of translating the divergence "
						  "away.");
				storageRefAliasTargets[decl] = resolved->first;
				Json result = Json::object();
				result["kind"] = "block";
				result["statements"] = Json::array();
				for (Json const& snapshotLet: resolved->second)
					result["statements"].emplace_back(snapshotLet);
				return result;
			}
			if (auto reference = exportStorageRefValue(_rhs))
			{
				storageRefValueLocals.insert(decl);
				Json result = Json::object();
				result["kind"] = "let";
				result["sourceDeclarationId"] = Json();
				result["name"] = decl->name().empty() ? identifier->name() : decl->name();
				result["type"] = storageRefWireType(decl->annotation().type);
				result["value"] = std::move(*reference);
				return result;
			}
			if (storageRefWriteThroughLocals.count(decl))
				throw UnsupportedSolCore(
					"Write through the untracked storage-pointer local `" + decl->name()
					+ "` is not "
					  "modeled: its single binding assignment does not resolve to a storage path, so "
					  "the local lowers to a copy of the pointed-to VALUE and a later write through it "
					  "would be exported as a rebind of that dead copy — leaving storage unchanged "
					  "while the real EVM writes through the pointer. Refusing instead of silently "
					  "dropping the state mutation.");
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
		if (baseType
			&& (baseType->category() == Type::Category::Array || baseType->category() == Type::Category::FixedBytes))
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
			if (auto const* baseIdent = dynamic_cast<Identifier const*>(&indexAccess->baseExpression());
				baseIdent
				&& !isTrackedStorageRefAlias(
					dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration)))
			{
				auto const* decl
					= dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
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
				requireLoweredStorageWriteRootOrThrow(_lhs, "array element write");
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
			// Struct-nested mapping assignment: config.targetAmts[k] = value
			// (the mapping index-access base is a struct-field MemberAccess
			// chain rooted at a state variable, not a bare mapping
			// Identifier). Tried before the mapping-of-mapping case below
			// since the two shapes never overlap: exportStorageMapLValueDeep
			// requires at least one struct-field hop, which a real
			// `outer[k1][k2]` mapping-of-mapping base (an IndexAccess, not a
			// MemberAccess) never has.
			try
			{
				auto [path, key] = exportStorageMapLValueDeep(_lhs);
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
			}
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
				//
				// [storage-ref-alias review fix] A COMPOUND assignment
				// (`m[k] += v` etc.) reaching this catch must expand to the
				// read-modify-write value, exactly like the two try-paths
				// above do via compoundValue. Before this fix the catch
				// passed the raw `rhsJson` through, silently turning
				// `base[k] += v` into `base[k] = v` — concretely observed
				// on StRSRP1._transfer's `eraStakes[to] += amount` once the
				// general storage-ref alias substitution made this path
				// reachable for alias-based mapping writes (an alias base
				// identifier resolves to a rooted storage_map_get read, so
				// exportStorageMapLValue throws and both try-paths above
				// fail, landing here). A compound op with no index
				// expression cannot be expanded — fail closed instead of
				// mis-lowering.
				requireLoweredStorageWriteRootOrThrow(_lhs, "mapping/index write");
				Json effectiveRhs = rhsJson;
				if (_op != Token::Assign)
				{
					if (!indexAccess->indexExpression())
						throw UnsupportedSolCore("Compound index assignment without index expression.");
					Json current = Json::object();
					current["kind"] = "array_get";
					current["base"] = exportExpr(indexAccess->baseExpression());
					current["index"] = exportExpr(*indexAccess->indexExpression());
					effectiveRhs = compoundValue(current, rhsJson);
				}
				Json result = Json::object();
				result["kind"] = "expr";
				Json callExpr = Json::object();
				callExpr["kind"] = "internal_call";
				callExpr["function"] = "array_set_expr";
				callExpr["args"] = Json::array();
				callExpr["args"].emplace_back(exportExpr(indexAccess->baseExpression()));
				if (indexAccess->indexExpression())
					callExpr["args"].emplace_back(exportExpr(*indexAccess->indexExpression()));
				callExpr["args"].emplace_back(effectiveRhs);
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

		auto exportStructUpdate = [&](Expression const& baseExpr)
		{
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
			// A tracked, resolved storage-ref alias must NOT take the
			// "local struct member write" branch below: that branch
			// reassigns the LOCAL POINTER variable to a new struct value
			// and discards it, silently dropping the real storage write
			// (design §3.3's documented write-dropping bug for this exact
			// shape). Falling through to the generic struct_update-as-expr
			// path lets exportStructUpdate's own exportExpr(baseExpr) call
			// substitute the alias's rooted read, producing a
			// StorageGet/StorageMapGet-rooted Expr(StructUpdate(...)) the
			// existing writeback pass (Base.ml's
			// lower_nested_storage_array_set / the OCaml frontend) already
			// re-roots into a real RMW storage write.
			if (decl && !decl->isStateVariable() && !isTrackedStorageRefAlias(decl))
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
		requireLoweredStorageWriteRootOrThrow(_lhs, "struct field write");
		Json result = Json::object();
		result["kind"] = "expr";
		result["value"] = exportStructUpdate(memberAccess->expression());
		return result;
	}

	// Tuple LHS: (a, b) = (1, 2) — export as a block of individual assignments
	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_lhs))
	{
		for (auto const& component: tuple->components())
			if (component)
				requireLoweredStorageWriteRootOrThrow(*component, "tuple assignment component");
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
		requireLoweredStorageWriteRootOrThrow(_lhs, "generic assignment");
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
	auto mkDirectStorage = [&](std::string const& _field, Json const& _storedValue)
	{
		Json result = Json::object();
		result["kind"] = "storage_set";
		result["field"] = _field;
		result["value"] = _storedValue;
		return result;
	};

	auto mkDirectMap = [&](std::vector<std::string> const& _path, Json const& _key, Json const& _storedValue)
	{
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
		// EIP-1153 `transient` state variable: dedicated statement, never
		// storage_set (see exportAssignment's twin arm).
		if (isTransientStateVar(decl))
		{
			Json result = Json::object();
			result["kind"] = "transient_set";
			result["field"] = decl->name();
			result["value"] = _value;
			return result;
		}
		if (decl->immutable())
			return immutableSet(*decl, _value);
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
		if (baseType
			&& (baseType->category() == Type::Category::Array || baseType->category() == Type::Category::FixedBytes))
		{
			if (!indexAccess->indexExpression())
				throw UnsupportedSolCore("Array index assignment without index.");
			if (auto const* baseIdent = dynamic_cast<Identifier const*>(&indexAccess->baseExpression());
				baseIdent
				&& !isTrackedStorageRefAlias(
					dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration)))
			{
				auto const* decl
					= dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
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
			requireLoweredStorageWriteRootOrThrow(_lhs, "array element write");
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
			// Struct-nested mapping lvalue: config.targetAmts[k] (this is
			// exactly what `delete config.targetAmts[k];` and the identical
			// plain assignment both need). See exportStorageMapLValueDeep's
			// doc comment for why this is write-path-only and why it cannot
			// shadow the mapping-of-mapping case below.
			try
			{
				auto [path, key] = exportStorageMapLValueDeep(_lhs);
				return mkDirectMap(path, key, _value);
			}
			catch (...)
			{
			}
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
				requireLoweredStorageWriteRootOrThrow(_lhs, "mapping/index write");
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

		auto exportStructUpdate = [&](Expression const& baseExpr)
		{
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
			if (decl && !decl->isStateVariable() && !isTrackedStorageRefAlias(decl))
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
		requireLoweredStorageWriteRootOrThrow(_lhs, "struct field write");
		Json result = Json::object();
		result["kind"] = "expr";
		result["value"] = exportStructUpdate(memberAccess->expression());
		return result;
	}

	if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_lhs))
	{
		for (auto const& component: tuple->components())
			if (component)
				requireLoweredStorageWriteRootOrThrow(*component, "tuple assignment component");
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

	requireLoweredStorageWriteRootOrThrow(_lhs, "generic assignment");
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
	// `++`/`--` are the same "silently unsigned" bug class as compound
	// assignment (SolCore design §3-2): they share none of
	// `compoundAssignmentValue`'s logic (this is a sibling function, not a
	// caller of it), so the signedness check has to be repeated here too.
	auto bits = signedOperationBits(_targetType);
	switch (_op)
	{
	case Token::Inc:
		if (bits)
		{
			result["kind"] = "i256_add";
			result["bits"] = static_cast<int>(*bits);
			markUncheckedContext(result);
		}
		else
		{
			result["kind"] = "u256_add";
			markUncheckedContext(result);
			tagUnsignedArithWidth(result, _targetType);
		}
		break;
	case Token::Dec:
		if (bits)
		{
			result["kind"] = "i256_sub";
			result["bits"] = static_cast<int>(*bits);
			markUncheckedContext(result);
		}
		else
		{
			result["kind"] = "u256_sub";
			markUncheckedContext(result);
			tagUnsignedArithWidth(result, _targetType);
		}
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

// A resolved-Type* mirror of exportTypeName(TypeName const&) (~1241), needed
// because `delete` targets only carry annotation().type -- there is no
// TypeName AST node to walk for e.g. `delete config.erc20s;`. Kept in sync
// with exportTypeName's integer-width table and its Bytes/String -> "array
// of u8" special case: type decls and value expressions must monomorphize
// to the same `new_array__<mangled>` runtime helper name, so any divergence
// here would be a silent-mismatch bug, not just a style difference.
// Fails closed (throws UnsupportedSolCore) on any category it does not
// explicitly recognize, per [SolCore audit finding #9]: no zero-substitution.
Json exportResolvedType(Type const* _type, bool _storage)
{
	if (!_type)
		throw UnsupportedSolCore("Unresolved type in SolCore exporter.");
	bool storage = _storage || _type->category() == Type::Category::Mapping;
	if (auto const* reference = dynamic_cast<ReferenceType const*>(_type))
		storage = storage || reference->location() == DataLocation::Storage;

	switch (_type->category())
	{
	case Type::Category::Address:
	case Type::Category::Contract:
		return Json("address");
	case Type::Category::Bool:
		return Json("bool");
	case Type::Category::Integer:
	{
		auto const* intType = dynamic_cast<IntegerType const*>(_type);
		unsigned bits = intType->numBits();
		if (!intType->isSigned())
		{
			switch (bits)
			{
			case 8:
				return Json("u8");
			case 16:
				return Json("u16");
			case 32:
				return Json("u32");
			case 64:
				return Json("u64");
			case 128:
				return Json("u128");
			case 256:
				return Json("u256");
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
		switch (bits)
		{
		case 8:
			return Json("i8");
		case 16:
			return Json("i16");
		case 32:
			return Json("i32");
		case 64:
			return Json("i64");
		case 128:
			return Json("i128");
		case 256:
			return Json("i256");
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
	case Type::Category::FixedBytes:
	{
		auto const* fbType = dynamic_cast<FixedBytesType const*>(_type);
		return Json("bytes" + std::to_string(fbType->numBytes()));
	}
	case Type::Category::Enum:
	{
		auto const* enumType = dynamic_cast<EnumType const*>(_type);
		Json result = Json::object();
		result["kind"] = "enum";
		result["name"] = exportedEnumName(enumType->enumDefinition());
		return result;
	}
	case Type::Category::Struct:
	{
		auto const* structType = dynamic_cast<StructType const*>(_type);
		Json result = Json::object();
		result["kind"] = storage ? "storage_named" : "named";
		result["name"] = exportedStructName(structType->structDefinition());
		return result;
	}
	case Type::Category::Array:
	{
		auto const* arrayType = dynamic_cast<ArrayType const*>(_type);
		// Dynamic bytes/string is represented on the wire as an array of u8,
		// mirroring exportTypeName's Token::Bytes/Token::String arms -- this
		// is what makes `new bytes(0)`-shaped and delete-shaped u8-array
		// values monomorphize to the same `new_array__u8` helper as an
		// actual `bytes`/`string` declaration.
		Json element
			= arrayType->isByteArrayOrString() ? Json("u8") : exportResolvedType(arrayType->baseType(), storage);
		Json result = Json::object();
		if (arrayType->isDynamicallySized())
		{
			result["kind"] = storage ? "storage_array" : "array";
			result["element"] = element;
		}
		else
		{
			u256 len = arrayType->length();
			if (len > std::numeric_limits<unsigned long>::max())
				throw UnsupportedSolCore("Fixed-size array length out of range in SolCore exporter.");
			result["kind"] = storage ? "storage_fixed_array" : "fixed_array";
			result["element"] = element;
			result["size"] = len.convert_to<unsigned long>();
		}
		return result;
	}
	case Type::Category::Tuple:
	{
		auto const* tupleType = dynamic_cast<TupleType const*>(_type);
		Json result = Json::object();
		result["kind"] = "tuple";
		result["elements"] = Json::array();
		for (size_t i = 0; i < tupleType->components().size(); ++i)
		{
			Type const* component = tupleType->components()[i];
			if (!component)
				throw UnsupportedSolCore(
					"Resolved tuple type contains an omitted placeholder component at index " + std::to_string(i) + "."
				);
			result["elements"].emplace_back(exportResolvedType(component, storage));
		}
		return result;
	}
	case Type::Category::Mapping:
	{
		auto const* mappingType = dynamic_cast<MappingType const*>(_type);
		Json result = Json::object();
		result["kind"] = "mapping";
		result["key"] = exportResolvedType(mappingType->keyType(), false);
		result["value"] = exportResolvedType(mappingType->valueType(), true);
		return result;
	}
	case Type::Category::UserDefinedValueType:
	{
		auto const* udvt = dynamic_cast<UserDefinedValueType const*>(_type);
		return exportResolvedType(&udvt->underlyingType(), storage);
	}
	case Type::Category::Function:
	{
		auto const* fnType = dynamic_cast<FunctionType const*>(_type);
		if (fnType && fnType->kind() == FunctionType::Kind::External)
		{
			Json result = Json::object();
			result["kind"] = "external_function";
			return result;
		}
		if (fnType && fnType->kind() == FunctionType::Kind::Internal)
		{
			auto component = [](Type const* _componentType)
			{
				Json item = Json::object();
				bool componentStorage = _componentType && _componentType->category() == Type::Category::Mapping;
				if (auto const* refType = dynamic_cast<ReferenceType const*>(_componentType))
				{
					item["location"] = dataLocationName(refType->location());
					componentStorage = refType->location() == DataLocation::Storage;
				}
				else if (componentStorage)
					item["location"] = "storage";
				else
					item["location"] = "none";
				item["type"] = exportResolvedType(_componentType, componentStorage);
				return item;
			};
			Json result = Json::object();
			result["kind"] = "internal_function";
			result["params"] = Json::array();
			for (Type const* paramType: fnType->parameterTypes())
				result["params"].emplace_back(component(paramType));
			result["returns"] = Json::array();
			for (Type const* returnType: fnType->returnParameterTypes())
				result["returns"].emplace_back(component(returnType));
			switch (fnType->stateMutability())
			{
			case StateMutability::Pure:
				result["mutability"] = "pure";
				break;
			case StateMutability::View:
				result["mutability"] = "view";
				break;
			case StateMutability::Payable:
				result["mutability"] = "payable";
				break;
			default:
				result["mutability"] = "nonpayable";
				break;
			}
			(void) registerInternalFnTable(*fnType, result);
			return result;
		}
		break;
	}
	default:
		break;
	}
	throw UnsupportedSolCore(
		"Unsupported resolved type in SolCore exporter: " +
		_type->humanReadableName() + " (category " +
		std::to_string(static_cast<int>(_type->category())) + ")");
}

Json typedDefaultValueForResolvedType(Type const* _type, bool _storage = false)
{
	Json result = Json::object();
	result["kind"] = "typed_default";
	Json resolvedType = exportResolvedType(_type, _storage);
	if (!_storage)
		if (auto const* reference = dynamic_cast<ReferenceType const*>(_type);
			reference && reference->location() == DataLocation::Memory)
		{
			Json memoryRef = Json::object();
			memoryRef["kind"] = "memory_ref";
			memoryRef["referent"] = std::move(resolvedType);
			resolvedType = std::move(memoryRef);
		}
	result["type"] = std::move(resolvedType);
	return result;
}

// `delete x;` resets `x` to its compiler-resolved default value. Keep that
// type authoritative in the wire node: scalar zero, enum-first, and
// dynamic-array-empty are consumer-side recursive default semantics. Arrays
// containing nested mappings remain refused because the EVM preserves their
// mapping data across delete-then-push, which no value-level default models.
Json deleteDefaultValueForResolvedType(Type const* _type)
{
	if (!_type)
		throw UnsupportedSolCore("delete on a target with unresolved type is unsupported.");

	if (auto const* udvt = dynamic_cast<UserDefinedValueType const*>(_type))
		return deleteDefaultValueForResolvedType(&udvt->underlyingType());

	if (exportSimpleType(*_type))
		return typedDefaultValueForResolvedType(_type);

	if (_type->category() == Type::Category::Enum)
		return typedDefaultValueForResolvedType(_type);

	if (_type->category() == Type::Category::Array)
	{
		auto const* arrayType = dynamic_cast<ArrayType const*>(_type);
		if (arrayType->containsNestedMapping())
			throw UnsupportedSolCore(
				"delete on an array whose element type contains a nested mapping is unsupported "
				"(the EVM itself preserves that mapping data across a delete-then-push, so no "
				"value-level default can soundly model it)");
		return typedDefaultValueForResolvedType(arrayType, true);
	}

	if (_type->category() == Type::Category::Mapping)
		throw UnsupportedSolCore(
			"delete on a mapping-typed target is unsupported (unreachable from valid Solidity: "
			"the type checker rejects `delete` on a mapping-typed lvalue).");

	throw UnsupportedSolCore("delete on this target type is unsupported.");
}

// Builds a chained `struct_update` spine over `_base` (a JSON expression --
// either the exported lvalue snapshot at the top level, or a synthetic
// `{"kind":"field",...}` projection when recursing into a struct-typed
// member) that resets every NON-MAPPING member to its delete-default,
// leaving mapping members untouched. This mirrors
// YulUtilFunctions::clearStorageStructFunction exactly: same declared
// member order, same "skip mapping members" rule, recursing into
// struct-typed members. Soundness: every leaf value is a constant (no
// cross-member reads), so member order is irrelevant, and non-updated
// (mapping) members flow through unchanged from `_base` -- exactly the
// EVM's own per-member clear-with-mapping-skip.
Json exportDeleteStructSpine(Json const& _base, StructDefinition const& _structDef)
{
	Json cur = _base;
	for (ASTPointer<VariableDeclaration> const& memberPtr: _structDef.members())
	{
		VariableDeclaration const& member = *memberPtr;
		Type const* memberType = member.type();
		if (!memberType)
			throw UnsupportedSolCore("delete: struct member '" + member.name() + "' has unresolved type.");
		if (memberType->category() == Type::Category::Mapping)
			continue; // preserved -- matches clearStorageStructFunction's mapping-member skip

		Json fieldRead = Json::object();
		fieldRead["kind"] = "field";
		fieldRead["base"] = cur;
		fieldRead["field"] = member.name();

		Json value;
		if (memberType->category() == Type::Category::Struct)
		{
			auto const* memberStructType = dynamic_cast<StructType const*>(memberType);
			value = exportDeleteStructSpine(fieldRead, memberStructType->structDefinition());
		}
		else
			value = deleteDefaultValueForResolvedType(memberType);

		Json update = Json::object();
		update["kind"] = "struct_update";
		update["base"] = cur;
		update["field"] = member.name();
		update["value"] = value;
		cur = update;
	}
	return cur;
}

Json exportDelete(Expression const& _target)
{
	Type const* type = _target.annotation().type;
	if (!type)
		throw UnsupportedSolCore("delete on a target with unresolved type is unsupported.");
	if (auto const* udvt = dynamic_cast<UserDefinedValueType const*>(type))
		type = &udvt->underlyingType();

	if (auto const* referenceType = dynamic_cast<ReferenceType const*>(type);
		referenceType && referenceType->location() == DataLocation::Memory)
	{
		auto const* identifier = dynamic_cast<Identifier const*>(&_target);
		auto const* declaration
			= identifier ? dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration)
						 : nullptr;
		if (!declaration || declaration->isStateVariable()
			|| declaration->referenceLocation() != VariableDeclaration::Location::Memory)
			throw UnsupportedSolCore("delete on a memory reference requires a resolved local memory-view location.");
		Json result = Json::object();
		result["kind"] = "expr";
		Json call = Json::object();
		call["kind"] = "internal_call";
		call["function"] = "memory_delete";
		call["args"] = Json::array();
		call["args"].emplace_back(exportExpr(_target));
		result["value"] = std::move(call);
		return result;
	}
	if (hasResidualStorageRefRoot(_target))
	{
		auto reference = exportStorageRefValue(_target);
		if (!reference)
			throw UnsupportedSolCore(
				"delete through a storage-reference parameter did not resolve to an exact typed path.");
		std::string tempName = "__solcore_delete_sref_" + std::to_string(stableSyntheticNodeId(_target));
		Json letReference = Json::object();
		letReference["kind"] = "let";
		letReference["sourceDeclarationId"] = Json();
		letReference["name"] = tempName;
		letReference["type"] = storageRefWireType(type);
		letReference["value"] = std::move(*reference);

		Json localReference = localExpr(tempName);
		Json deletedValue;
		if (hasSlotPreservingDynamicStorageArraySemantics(type))
		{
			deletedValue["kind"] = "internal_call";
			deletedValue["function"] = "storage_array_clear";
			deletedValue["args"] = Json::array();
			deletedValue["args"].emplace_back(storageRefGetJson(localReference, type));
		}
		else if (type->category() == Type::Category::Struct)
		{
			auto const* structType = dynamic_cast<StructType const*>(type);
			deletedValue = exportDeleteStructSpine(
				storageRefGetJson(localReference, type), structType->structDefinition());
		}
		else
			deletedValue = deleteDefaultValueForResolvedType(type);

		Json set = Json::object();
		set["kind"] = "storage_ref_set";
		set["referentType"] = exportResolvedType(type, true);
		set["reference"] = std::move(localReference);
		set["value"] = std::move(deletedValue);

		Json block = Json::object();
		block["kind"] = "block";
		block["statements"] = Json::array();
		block["statements"].emplace_back(std::move(letReference));
		block["statements"].emplace_back(std::move(set));
		return block;
	}

	if (hasSlotPreservingDynamicStorageArraySemantics(type))
	{
		auto rootedTarget
			= exportResolvedStorageRefUse(_target, StorageRefKeySnapshotMode::DirectMappingKey);
		if (!rootedTarget)
			throw UnsupportedSolCore("delete on a dynamic storage array requires a resolved rooted storage lvalue.");

		Json result = Json::object();
		result["kind"] = "expr";
		Json call = Json::object();
		call["kind"] = "internal_call";
		call["function"] = "storage_array_clear";
		call["args"] = Json::array();
		call["args"].emplace_back(std::move(*rootedTarget));
		result["value"] = std::move(call);
		return result;
	}

	if (auto const* arrayType = dynamic_cast<ArrayType const*>(type);
		arrayType && arrayType->location() == DataLocation::Storage && arrayType->isDynamicallySized()
		&& arrayElementContainsRecursiveStruct(*arrayType))
		throw UnsupportedSolCore("delete on a recursive dynamic storage-array shape is unsupported.");

	if (type->category() == Type::Category::Struct)
	{
		auto const* structType = dynamic_cast<StructType const*>(type);
		StructDefinition const& structDef = structType->structDefinition();
		bool allMembersAreMappings = std::all_of(
			structDef.members().begin(),
			structDef.members().end(),
			[](ASTPointer<VariableDeclaration> const& _member)
			{
				Type const* memberType = _member->type();
				return memberType && memberType->category() == Type::Category::Mapping;
			});
		if (allMembersAreMappings)
		{
			// Every member is preserved (mapping-skip): the delete is a
			// genuine no-op, so emit exactly that rather than a spurious
			// self-assignment.
			Json block = Json::object();
			block["kind"] = "block";
			block["statements"] = Json::array();
			return block;
		}
		Json spine = exportDeleteStructSpine(exportExpr(_target), structDef);
		return exportDirectAssignment(_target, spine);
	}

	return exportDirectAssignment(_target, deleteDefaultValueForResolvedType(type));
}

Json exportUnaryMutationReturn(UnaryOperation const& _unary)
{
	Expression const& target = _unary.subExpression();
	std::string tempName = "__solcore_tmp_" + std::to_string(stableSyntheticNodeId(_unary));

	Json block = Json::object();
	block["kind"] = "block";
	block["statements"] = Json::array();

	if (_unary.isPrefixOperation())
	{
		Json letStmt = Json::object();
		letStmt["kind"] = "let";
		letStmt["sourceDeclarationId"] = Json();
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
		letStmt["sourceDeclarationId"] = Json();
		letStmt["name"] = tempName;
		letStmt["type"] = "u256";
		letStmt["value"] = exportExpr(target);
		block["statements"].emplace_back(std::move(letStmt));
		block["statements"].emplace_back(exportDirectAssignment(
			target, mutationValue(localExpr(tempName), _unary.getOperator(), target.annotation().type)));
	}

	Json ret = Json::object();
	ret["kind"] = "return";
	ret["value"] = localExpr(tempName);
	block["statements"].emplace_back(std::move(ret));
	return block;
}

// === Hoisting of mutating sub-expressions (see HoistScope above) ===========

// Defined with the AST write-set oracle further below; reused here so the
// hoisting conflict scan resolves lvalue bases and virtual call targets with
// exactly the same rules as the independent fidelity oracle.
Identifier const* peelToBaseIdentifierForWriteOracle(Expression const& _expr);
FunctionDefinition const*
resolveWriteOracleCallTarget(Expression const& _callee, ContractDefinition const* _mostDerivedContract);

/// Builtin call kinds that can neither write persistent storage nor invoke
/// other code that could (no external calls, no storage array push/pop, no
/// contract creation, no value transfer). Reads of storage are impossible
/// for these too, so they are order-insensitive w.r.t. a hoisted write.
bool isHoistStorageInertCallKind(FunctionType::Kind _kind)
{
	switch (_kind)
	{
	case FunctionType::Kind::KECCAK256:
	case FunctionType::Kind::SHA256:
	case FunctionType::Kind::RIPEMD160:
	case FunctionType::Kind::ECRecover:
	case FunctionType::Kind::AddMod:
	case FunctionType::Kind::MulMod:
	case FunctionType::Kind::ABIEncode:
	case FunctionType::Kind::ABIEncodePacked:
	case FunctionType::Kind::ABIEncodeWithSelector:
	case FunctionType::Kind::ABIEncodeCall:
	case FunctionType::Kind::ABIEncodeWithSignature:
	case FunctionType::Kind::ABIDecode:
	case FunctionType::Kind::Event:
	case FunctionType::Kind::Error:
	case FunctionType::Kind::Assert:
	case FunctionType::Kind::Require:
	case FunctionType::Kind::Wrap:
	case FunctionType::Kind::Unwrap:
	case FunctionType::Kind::GasLeft:
	case FunctionType::Kind::BlockHash:
	case FunctionType::Kind::BlobHash:
	case FunctionType::Kind::BytesConcat:
	case FunctionType::Kind::StringConcat:
	case FunctionType::Kind::ObjectCreation: // `new T[](n)` memory allocation
	case FunctionType::Kind::MetaType:
	case FunctionType::Kind::ERC7201:
		return true;
	default:
		return false;
	}
}

enum class HoistCallClass
{
	Inert,		  ///< cannot read or write persistent storage at all
	InternalView, ///< internal view/pure callee — safe iff its transitive body never touches the mutated variable
	Blocking	  ///< anything else (external, unknown, state-mutating) — conflicts
};

HoistCallClass classifyCallForHoistScan(FunctionCall const& _call, FunctionDefinition const*& _calleeOut)
{
	_calleeOut = nullptr;
	if (*_call.annotation().kind == FunctionCallKind::TypeConversion
		|| *_call.annotation().kind == FunctionCallKind::StructConstructorCall)
		return HoistCallClass::Inert;
	auto const* funType = dynamic_cast<FunctionType const*>(_call.expression().annotation().type);
	if (!funType)
		return HoistCallClass::Blocking;
	if (isHoistStorageInertCallKind(funType->kind()))
		return HoistCallClass::Inert;
	if (funType->kind() == FunctionType::Kind::Internal)
	{
		FunctionDefinition const* callee = resolveWriteOracleCallTarget(_call.expression(), activeExportContract);
		if (!callee || !callee->isImplemented())
			return HoistCallClass::Blocking;
		if (callee->stateMutability() != StateMutability::View && callee->stateMutability() != StateMutability::Pure)
			return HoistCallClass::Blocking;
		_calleeOut = callee;
		return HoistCallClass::InternalView;
	}
	return HoistCallClass::Blocking;
}

bool viewCalleeCannotTouchDecl(
	FunctionDefinition const& _callee,
	VariableDeclaration const& _decl,
	std::set<FunctionDefinition const*>& _visited,
	int _depth);

/// Scans an internal view/pure callee's body for anything that could make
/// the callee's result (or behavior) depend on the hoisted-over variable:
/// a direct reference to it, any storage-pointer local (may alias it), any
/// inline assembly (opaque sload), or any call that is not itself provably
/// inert. `safe` stays true only when none of those occur.
struct HoistCalleeBodyScanner: ASTConstVisitor
{
	HoistCalleeBodyScanner(VariableDeclaration const& _decl, std::set<FunctionDefinition const*>& _visited, int _depth)
		: decl(_decl), visited(_visited), depth(_depth)
	{
	}

	VariableDeclaration const& decl;
	std::set<FunctionDefinition const*>& visited;
	int depth;
	bool safe = true;

	bool visit(Identifier const& _identifier) override
	{
		Declaration const* referenced = _identifier.annotation().referencedDeclaration;
		if (referenced == &decl)
			safe = false;
		else if (auto const* varDecl = dynamic_cast<VariableDeclaration const*>(referenced))
			// A storage-pointer local inside the callee could alias the
			// mutated variable's slots — fail closed.
			if (!varDecl->isStateVariable() && varDecl->referenceLocation() == VariableDeclaration::Location::Storage)
				safe = false;
		return safe;
	}

	bool visit(MemberAccess const& _memberAccess) override
	{
		// Qualified access to the mutated variable (e.g. `Base.stateVar`).
		if (_memberAccess.annotation().referencedDeclaration == &decl)
			safe = false;
		return safe;
	}

	bool visit(InlineAssembly const&) override
	{
		safe = false;
		return false;
	}

	bool visit(FunctionCall const& _call) override
	{
		if (!safe)
			return false;
		FunctionDefinition const* callee = nullptr;
		switch (classifyCallForHoistScan(_call, callee))
		{
		case HoistCallClass::Inert:
			break;
		case HoistCallClass::InternalView:
			if (!viewCalleeCannotTouchDecl(*callee, decl, visited, depth - 1))
				safe = false;
			break;
		case HoistCallClass::Blocking:
			safe = false;
			break;
		}
		return safe;
	}
};

bool viewCalleeCannotTouchDecl(
	FunctionDefinition const& _callee,
	VariableDeclaration const& _decl,
	std::set<FunctionDefinition const*>& _visited,
	int _depth)
{
	if (_depth <= 0)
		return false;
	if (!_visited.insert(&_callee).second)
		// Already scanned (or being scanned) in this analysis: a cycle
		// introduces no references beyond what its own scan covers.
		return true;
	if (!_callee.isImplemented())
		return false;

	// Modifiers execute as part of the callee; scan their resolved bodies
	// and invocation arguments too.
	for (auto const& modifierInvocation: _callee.modifiers())
	{
		ModifierDefinition const* modifierDefinition = resolveModifierDefinition(_callee, *modifierInvocation);
		if (!modifierDefinition || !modifierDefinition->isImplemented())
			return false;
		HoistCalleeBodyScanner modifierScanner(_decl, _visited, _depth);
		modifierDefinition->body().accept(modifierScanner);
		if (modifierInvocation->arguments())
			for (auto const& arg: *modifierInvocation->arguments())
				arg->accept(modifierScanner);
		if (!modifierScanner.safe)
			return false;
	}

	HoistCalleeBodyScanner scanner(_decl, _visited, _depth);
	_callee.body().accept(scanner);
	return scanner.safe;
}

/// Scans one once-evaluated header expression of the enclosing statement
/// for accesses that would make the hoisted write's position observable:
/// - any reference to the mutated base variable OUTSIDE the mutation
///   expression itself (a re-ordered read/write of the same location);
/// - in strict (storage-backed) mode: any storage-pointer identifier
///   (potential alias) outside the mutation, and any call anywhere in the
///   statement that is not storage-inert or a provably-unrelated internal
///   view/pure function;
/// - in non-strict (plain local) mode, calls inside the MUTATION subtree
///   still need the inert/view rule: the emitted read+write pair evaluates
///   the lvalue subtree twice, so calls in it must be duplication-safe.
struct HoistRootConflictScanner: ASTConstVisitor
{
	HoistRootConflictScanner(Expression const& _mutation, VariableDeclaration const& _decl, bool _strict)
		: mutation(_mutation), decl(_decl), strict(_strict)
	{
	}

	Expression const& mutation;
	VariableDeclaration const& decl;
	bool strict;
	int insideMutation = 0;
	bool conflict = false;
	bool foundMutation = false;
	std::set<FunctionDefinition const*> visited;

	bool visit(UnaryOperation const& _unary) override
	{
		if (&_unary == &mutation)
		{
			++insideMutation;
			foundMutation = true;
		}
		return !conflict;
	}
	void endVisit(UnaryOperation const& _unary) override
	{
		if (&_unary == &mutation)
			--insideMutation;
	}

	bool visit(Assignment const& _assignment) override
	{
		if (&_assignment == &mutation)
		{
			++insideMutation;
			foundMutation = true;
		}
		return !conflict;
	}
	void endVisit(Assignment const& _assignment) override
	{
		if (&_assignment == &mutation)
			--insideMutation;
	}

	bool visit(Identifier const& _identifier) override
	{
		if (insideMutation == 0)
		{
			Declaration const* referenced = _identifier.annotation().referencedDeclaration;
			if (referenced == &decl)
				conflict = true;
			else if (strict)
				if (auto const* varDecl = dynamic_cast<VariableDeclaration const*>(referenced))
					if (!varDecl->isStateVariable()
						&& varDecl->referenceLocation() == VariableDeclaration::Location::Storage)
						conflict = true;
		}
		return !conflict;
	}

	bool visit(MemberAccess const& _memberAccess) override
	{
		if (insideMutation == 0 && _memberAccess.annotation().referencedDeclaration == &decl)
			conflict = true;
		return !conflict;
	}

	bool visit(FunctionCall const& _call) override
	{
		if (conflict)
			return false;
		// Calls cannot observe or mutate a plain stack local, so in
		// non-strict mode only calls inside the (duplicated) mutation
		// subtree need checking.
		if (!strict && insideMutation == 0)
			return true;
		FunctionDefinition const* callee = nullptr;
		switch (classifyCallForHoistScan(_call, callee))
		{
		case HoistCallClass::Inert:
			break;
		case HoistCallClass::InternalView:
			// A pure/view callee can never reach the caller's PLAIN STACK
			// local (frame-private) and performs no state writes, so its
			// once-hoisted evaluation is observation-equivalent regardless
			// of its body (witness: solady `lnWad(w = lnWad(w))`, whose
			// assembly-bodied pure callee defeated the body walk). The body
			// walk remains required in strict mode, where `decl` is state
			// or storage-located and a view callee could READ it.
			if (strict && !viewCalleeCannotTouchDecl(*callee, decl, visited, 8))
				conflict = true;
			break;
		case HoistCallClass::Blocking:
			conflict = true;
			break;
		}
		return !conflict;
	}
};

std::string hoistTempName(ASTNode const& _node)
{
	return "__solcore_hoist_" + std::to_string(stableSyntheticNodeId(_node));
}

void requireHoistScopeAllowed()
{
	if (!activeHoistScope || !activeHoistScope->allowed)
		throw UnsupportedSolCore(
			"Mutating sub-expression (++/--/assignment used as a value) in a "
			"repeatedly-evaluated or otherwise unrecognized position (for "
			"example a loop condition or outside statement context) cannot be "
			"soundly hoisted to statement level.");
}

/// Throws UnsupportedSolCore unless hoisting `_mutation` (whose lvalue is
/// `_lvalue`) out of the current statement is observation-equivalent for
/// every intra-statement evaluation order (see the HoistScope comment for
/// the soundness argument). Must be called BEFORE anything is appended to
/// the hoist buffer so a refusal leaves no partial state behind.
void checkHoistConflictsOrThrow(Expression const& _mutation, Expression const& _lvalue)
{
	HoistScope& scope = *activeHoistScope;

	Identifier const* baseIdent = peelToBaseIdentifierForWriteOracle(_lvalue);
	if (!baseIdent)
		throw UnsupportedSolCore(
			"Mutating sub-expression whose target has no identifiable base "
			"variable cannot be hoisted soundly.");
	auto const* decl = dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
	if (!decl)
		throw UnsupportedSolCore("Mutating sub-expression targets something that is not a variable.");

	bool strict = decl->isStateVariable() || decl->referenceLocation() == VariableDeclaration::Location::Storage
				  || namespacedStorageAliases.count(decl->name()) > 0;

	bool found = false;
	for (Expression const* root: scope.roots)
	{
		HoistRootConflictScanner scanner(_mutation, *decl, strict);
		root->accept(scanner);
		if (scanner.conflict)
			throw UnsupportedSolCore(
				"Mutating sub-expression cannot be soundly hoisted: the "
				"enclosing statement contains another access that may read or "
				"write the mutated location, so intra-statement evaluation "
				"order (unspecified in Solidity) would become observable.");
		if (scanner.foundMutation)
			found = true;
	}
	if (!found)
		// Safety net: the mutation is being exported from a position that
		// is not part of the registered once-evaluated statement header —
		// an unaudited context. Fail closed.
		throw UnsupportedSolCore(
			"Mutating sub-expression in an unrecognized expression position "
			"cannot be hoisted.");
}

Json exportHoistedUnaryMutation(UnaryOperation const& _unary)
{
	Expression const& target = _unary.subExpression();

	if (activeHoistScope)
	{
		auto it = activeHoistScope->memo.find(static_cast<int64_t>(_unary.id()));
		if (it != activeHoistScope->memo.end())
			// This exact mutation node was already hoisted during this
			// statement's export (compound-assignment lvalues and
			// exception-based lowering fallbacks re-export sub-expressions):
			// reuse the temp, never duplicate the side effect.
			return localExpr(it->second);
	}

	requireHoistScopeAllowed();
	checkHoistConflictsOrThrow(_unary, target);

	std::string tempName = hoistTempName(_unary);
	Json letStmt = Json::object();
	letStmt["kind"] = "let";
	letStmt["sourceDeclarationId"] = Json();
	letStmt["name"] = tempName;
	letStmt["type"] = "u256";
	Json writeStmt;
	if (_unary.isPrefixOperation())
	{
		// ++x / --x: the value is the POST-mutation value.
		letStmt["value"] = mutationValue(exportExpr(target), _unary.getOperator(), target.annotation().type);
		writeStmt = exportDirectAssignment(target, localExpr(tempName));
	}
	else
	{
		// x++ / x--: the value is the PRE-mutation value.
		letStmt["value"] = exportExpr(target);
		writeStmt = exportDirectAssignment(
			target, mutationValue(localExpr(tempName), _unary.getOperator(), target.annotation().type));
	}

	// Append + memoize only now, after every piece lowered successfully:
	// a throw above must leave no partial hoist state behind.
	activeHoistScope->statements.emplace_back(std::move(letStmt));
	activeHoistScope->statements.emplace_back(std::move(writeStmt));
	activeHoistScope->memo[static_cast<int64_t>(_unary.id())] = tempName;
	return localExpr(tempName);
}

Json exportHoistedAssignExpr(Assignment const& _assignment)
{
	Expression const& lhs = _assignment.leftHandSide();

	if (activeHoistScope)
	{
		auto it = activeHoistScope->memo.find(static_cast<int64_t>(_assignment.id()));
		if (it != activeHoistScope->memo.end())
			return localExpr(it->second);
	}

	requireHoistScopeAllowed();
	checkHoistConflictsOrThrow(_assignment, lhs);

	// The value of an assignment expression is the assigned value (the
	// combined value for compound operators), captured in a temp BEFORE the
	// write so no post-write re-read of the lvalue is needed.
	std::optional<Json> simpleType = lhs.annotation().type ? exportSimpleType(*lhs.annotation().type) : std::nullopt;
	if (!simpleType)
		throw UnsupportedSolCore(
			"Assignment of a non-word-sized value used as a sub-expression "
			"cannot be hoisted soundly.");

	Json rhsJson = exportExpr(_assignment.rightHandSide());
	Json valueJson;
	if (_assignment.assignmentOperator() == Token::Assign)
		valueJson = rhsJson;
	else
		valueJson = compoundAssignmentValue(
			_assignment.assignmentOperator(), lhs.annotation().type, exportExpr(lhs), rhsJson);

	std::string tempName = hoistTempName(_assignment);
	Json letStmt = Json::object();
	letStmt["kind"] = "let";
	letStmt["sourceDeclarationId"] = Json();
	letStmt["name"] = tempName;
	letStmt["type"] = *simpleType;
	letStmt["value"] = valueJson;
	Json writeStmt = exportDirectAssignment(lhs, localExpr(tempName));

	activeHoistScope->statements.emplace_back(std::move(letStmt));
	activeHoistScope->statements.emplace_back(std::move(writeStmt));
	activeHoistScope->memo[static_cast<int64_t>(_assignment.id())] = tempName;
	return localExpr(tempName);
}

bool inlineAssemblyOperationIs(
	std::string_view _operation,
	std::initializer_list<std::string_view> _candidates)
{
	return std::find(_candidates.begin(), _candidates.end(), _operation) != _candidates.end();
}

/// Collects the producer-owned interface of one inline assembly block.
///
/// The Yul walker is deliberately driven by the exact dialect attached to the
/// InlineAssembly AST node. Builtin handles are dialect-specific; resolving
/// them from source spellings would lose aliases and could classify an
/// operation according to a dialect that did not compile this block.
class InlineAssemblyInterfaceCollector: private yul::ASTWalker
{
public:
	explicit InlineAssemblyInterfaceCollector(InlineAssembly const& _assembly):
		m_dialect(_assembly.dialect()),
		m_externalReferences(_assembly.annotation().externalReferences)
	{
		yul::Block const& root = _assembly.operations().root();
		yul::forEach<yul::FunctionDefinition const>(root, [&](yul::FunctionDefinition const& _function) {
			m_localFunctions.insert(_function.name.str());
		});

		// OpenZeppelin SafeERC20 v5.4 is the motivating witness: _safeTransfer
		// materializes transfer.selector and its two ABI words with mstore,
		// then issues call(gas(), token, 0, 0, 0x44, 0, 0x20).
		// Preserve that exact producer fact only; every uncertain shape is
		// intentionally represented by absence from staticExternalCalls.
		collectStaticExternalCalls(root);

		// ASTWalker recursively visits nested blocks, conditions, switches,
		// loops, and local Yul function bodies. Assignment is overridden below
		// so its external identifiers receive the correct LHS/RHS access.
		yul::ASTWalker::operator()(root);

		for (auto const& [identifier, information]: m_externalReferences)
		{
			if (!identifier || !information.declaration)
				throw UnsupportedSolCore(
					"Inline assembly contains an unresolved compiler external reference.");
			if (!m_seenExternalReferences.count(identifier))
				throw UnsupportedSolCore(
					"Inline assembly external-reference metadata does not correspond "
					"to an identifier use in the compiler-owned Yul AST.");
		}
	}

	Json interfaceJson() const
	{
		Json result = Json::object();
		result["locals"] = Json::array();
		for (auto const& [key, access]: m_localAccesses)
		{
			auto const& [declarationId, name, suffix] = key;
			auto declaration = m_declarations.find(declarationId);
			if (declaration == m_declarations.end())
				throw UnsupportedSolCore(
					"Inline assembly local interface lost its declaration-owned carrier type.");

			Json row = Json::object();
			row["declarationId"] = declarationId;
			row["name"] = name;
			row["type"] = declaration->second.type;
			switch (access)
			{
			case ReadAccess:
				row["access"] = "read";
				break;
			case WriteAccess:
				row["access"] = "write";
				break;
			case ReadAccess | WriteAccess:
				row["access"] = "read_write";
				break;
			default:
				throw UnsupportedSolCore(
					"Inline assembly local interface contains an invalid access classification.");
			}
			row["suffix"] = suffix;
			result["locals"].emplace_back(std::move(row));
		}

		result["memoryReads"] = m_memoryReads;
		result["memoryWrites"] = m_memoryWrites;
		result["environmentReads"] = stringSetJson(m_environmentReads);
		result["effects"] = stringSetJson(m_effects);
		result["operations"] = stringSetJson(m_operations);
		result["staticExternalCalls"] = m_staticExternalCalls;
		return result;
	}

private:
	using yul::ASTWalker::operator();

	static constexpr unsigned ReadAccess = 1;
	static constexpr unsigned WriteAccess = 2;

	struct DeclarationInfo
	{
		VariableDeclaration const* declaration;
		Json type;
	};

	static Json stringSetJson(std::set<std::string> const& _values)
	{
		Json result = Json::array();
		for (std::string const& value: _values)
			result.emplace_back(value);
		return result;
	}

	struct MemoryPrefix
	{
		std::map<u256, yul::Expression const*> stores;
		bool unknownWrite = false;
	};

	static std::optional<u256> yulNumber(yul::Expression const& _expression)
	{
		auto const* literal = std::get_if<yul::Literal>(&_expression);
		if (
			!literal
			|| literal->kind != yul::LiteralKind::Number
			|| literal->value.unlimited())
			return std::nullopt;
		return literal->value.value();
	}

	static std::optional<std::string> selectorHex(u256 const& _selector)
	{
		if (_selector > 0xffffffff)
			return std::nullopt;
		static constexpr char hexDigits[] = "0123456789abcdef";
		std::string result(8, '0');
		uint32_t value = _selector.convert_to<uint32_t>();
		for (size_t i = 0; i < result.size(); ++i)
		{
			result[result.size() - i - 1] = hexDigits[value & 0xf];
			value >>= 4;
		}
		return result;
	}

	static std::optional<std::string> soliditySelectorHex(Expression const& _initializer)
	{
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_initializer))
			if (memberAccess->memberName() == "selector")
				if (auto const* functionType =
					dynamic_cast<FunctionType const*>(memberAccess->expression().annotation().type))
				{
					try
					{
						std::string selector = functionType->externalIdentifierHex();
						if (selector.size() != 8)
							return std::nullopt;
						for (char& digit: selector)
						{
							if (digit >= 'A' && digit <= 'F')
								digit = static_cast<char>(digit - 'A' + 'a');
							else if (!((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f')))
								return std::nullopt;
						}
						return selector;
					}
					catch (...)
					{
						return std::nullopt;
					}
				}

		if (!dynamic_cast<Literal const*>(&_initializer))
			return std::nullopt;
		auto value = ConstantEvaluator::tryEvaluate(_initializer);
		if (!std::holds_alternative<rational>(value.value))
			return std::nullopt;
		rational const& number = std::get<rational>(value.value);
		if (number.denominator() != 1 || number.numerator() < 0 || number.numerator() > 0xffffffff)
			return std::nullopt;
		return selectorHex(u256(number.numerator().convert_to<uint32_t>()));
	}

	std::optional<std::string> externalLocalName(yul::Identifier const& _identifier) const
	{
		auto reference = m_externalReferences.find(&_identifier);
		if (reference == m_externalReferences.end() || !reference->second.suffix.empty())
			return std::nullopt;
		if (!dynamic_cast<VariableDeclaration const*>(reference->second.declaration))
			return std::nullopt;
		return _identifier.name.str();
	}

	std::optional<std::string> selectorWordHex(yul::Expression const& _expression) const
	{
		if (auto literal = yulNumber(_expression))
			return selectorHex(*literal);

		auto const* identifier = std::get_if<yul::Identifier>(&_expression);
		if (!identifier)
			return std::nullopt;
		auto reference = m_externalReferences.find(identifier);
		if (reference == m_externalReferences.end() || !reference->second.suffix.empty())
			return std::nullopt;
		auto const* variable =
			dynamic_cast<VariableDeclaration const*>(reference->second.declaration);
		auto const* fixedBytes =
			variable && variable->annotation().type
				? dynamic_cast<FixedBytesType const*>(variable->annotation().type)
				: nullptr;
		if (!fixedBytes || fixedBytes->numBytes() != 4 || !variable->value())
			return std::nullopt;
		return soliditySelectorHex(*variable->value());
	}

	bool isBuiltin(yul::FunctionCall const& _call, std::string_view _name) const
	{
		return yul::resolveBuiltinFunction(_call.functionName, m_dialect)
			&& yul::resolveFunctionName(_call.functionName, m_dialect) == _name;
	}

	bool isZeroLiteral(yul::Expression const& _expression) const
	{
		auto value = yulNumber(_expression);
		return value && *value == 0;
	}

	std::optional<std::string> maskedAddressRoot(yul::Expression const& _expression) const
	{
		auto const* andCall = std::get_if<yul::FunctionCall>(&_expression);
		if (!andCall || !isBuiltin(*andCall, "and") || andCall->arguments.size() != 2)
			return std::nullopt;
		auto const* root = std::get_if<yul::Identifier>(&andCall->arguments[0]);
		auto const* shrCall = std::get_if<yul::FunctionCall>(&andCall->arguments[1]);
		if (
			!root
			|| !shrCall
			|| !isBuiltin(*shrCall, "shr")
			|| shrCall->arguments.size() != 2)
			return std::nullopt;
		auto shift = yulNumber(shrCall->arguments[0]);
		auto const* notCall = std::get_if<yul::FunctionCall>(&shrCall->arguments[1]);
		if (
			!shift
			|| *shift != 96
			|| !notCall
			|| !isBuiltin(*notCall, "not")
			|| notCall->arguments.size() != 1
			|| !isZeroLiteral(notCall->arguments[0]))
			return std::nullopt;
		return externalLocalName(*root);
	}

	std::optional<Json> staticArgument(
		u256 const& _byteOffset,
		yul::Expression const& _expression) const
	{
		std::optional<std::string> source = maskedAddressRoot(_expression);
		std::string abiType = "address";
		if (!source)
		{
			auto const* identifier = std::get_if<yul::Identifier>(&_expression);
			if (!identifier)
				return std::nullopt;
			source = externalLocalName(*identifier);
			abiType = "uint256";
		}
		if (!source || _byteOffset > std::numeric_limits<uint64_t>::max())
			return std::nullopt;

		Json result = Json::object();
		result["byteOffset"] = _byteOffset.convert_to<uint64_t>();
		result["source"] = *source;
		result["abiType"] = std::move(abiType);
		return result;
	}

	void tryDecodeStaticExternalCall(
		yul::FunctionCall const& _call,
		MemoryPrefix const& _prefix)
	{
		std::string operation(yul::resolveFunctionName(_call.functionName, m_dialect));
		bool const isCall = operation == "call";
		bool const isStaticCall = operation == "staticcall";
		if (
			(!isCall && !isStaticCall)
			|| !yul::resolveBuiltinFunction(_call.functionName, m_dialect)
			|| _call.arguments.size() != (isCall ? 7 : 6)
			|| _prefix.unknownWrite)
			return;

		size_t const targetIndex = 1;
		size_t const valueIndex = 2;
		size_t const inputOffsetIndex = isCall ? 3 : 2;
		size_t const inputSizeIndex = isCall ? 4 : 3;
		auto const* targetIdentifier =
			std::get_if<yul::Identifier>(&_call.arguments[targetIndex]);
		auto target = targetIdentifier
			? externalLocalName(*targetIdentifier)
			: std::nullopt;
		auto inputOffset = yulNumber(_call.arguments[inputOffsetIndex]);
		auto inputSize = yulNumber(_call.arguments[inputSizeIndex]);
		if (
			!target
			|| (isCall && !isZeroLiteral(_call.arguments[valueIndex]))
			|| !inputOffset
			|| !inputSize
			|| *inputSize < 4
			|| (*inputSize - 4) % 32 != 0
			|| *inputOffset > std::numeric_limits<uint64_t>::max())
			return;
		u256 const inputEnd = *inputOffset + *inputSize;
		if (inputEnd < *inputOffset)
			return;

		auto selectorStore = _prefix.stores.find(*inputOffset);
		if (selectorStore == _prefix.stores.end())
			return;
		auto selector = selectorWordHex(*selectorStore->second);
		if (!selector)
			return;

		std::set<u256> expectedStoreOffsets{*inputOffset};
		Json arguments = Json::array();
		for (u256 offset = *inputOffset + 4; offset < inputEnd; offset += 32)
		{
			auto store = _prefix.stores.find(offset);
			if (store == _prefix.stores.end())
				return;
			auto argument = staticArgument(offset - *inputOffset, *store->second);
			if (!argument)
				return;
			expectedStoreOffsets.insert(offset);
			arguments.emplace_back(std::move(*argument));
		}

		for (auto const& [offset, value]: _prefix.stores)
		{
			(void)value;
			u256 const storeEnd = offset + 32;
			if (storeEnd < offset)
				return;
			if (
				offset < inputEnd
				&& storeEnd > *inputOffset
				&& !expectedStoreOffsets.count(offset))
				return;
		}

		Json result = Json::object();
		result["op"] = operation;
		result["target"] = *target;
		if (isCall)
			result["valueZero"] = true;
		result["selectorHex"] = *selector;
		result["args"] = std::move(arguments);
		result["argsComplete"] = true;
		m_staticExternalCalls.emplace_back(std::move(result));
	}

	bool scanExpressionForStaticCall(
		yul::Expression const& _expression,
		MemoryPrefix const& _prefix)
	{
		auto const* call = std::get_if<yul::FunctionCall>(&_expression);
		if (!call)
			return false;
		std::string operation(yul::resolveFunctionName(call->functionName, m_dialect));
		bool const isExternalCall =
			yul::resolveBuiltinFunction(call->functionName, m_dialect)
			&& (operation == "call" || operation == "staticcall");
		if (isExternalCall)
			tryDecodeStaticExternalCall(*call, _prefix);
		bool found = isExternalCall;
		for (yul::Expression const& argument: call->arguments)
			found = scanExpressionForStaticCall(argument, _prefix) || found;
		return found;
	}

	void recordMemoryStore(yul::Statement const& _statement, MemoryPrefix& _prefix)
	{
		auto const* expressionStatement = std::get_if<yul::ExpressionStatement>(&_statement);
		auto const* call = expressionStatement
			? std::get_if<yul::FunctionCall>(&expressionStatement->expression)
			: nullptr;
		if (!call)
			return;
		std::string operation(yul::resolveFunctionName(call->functionName, m_dialect));
		if (!yul::resolveBuiltinFunction(call->functionName, m_dialect))
			return;
		if (operation == "mstore")
		{
			if (call->arguments.size() != 2)
			{
				_prefix.unknownWrite = true;
				return;
			}
			auto offset = yulNumber(call->arguments[0]);
			if (!offset)
			{
				_prefix.unknownWrite = true;
				return;
			}
			_prefix.stores[*offset] = &call->arguments[1];
		}
		else if (inlineAssemblyOperationIs(
			operation,
			{"mstore8", "calldatacopy", "codecopy", "extcodecopy",
				"returndatacopy", "mcopy", "datacopy"}))
			_prefix.unknownWrite = true;
	}

	void collectStaticExternalCalls(yul::Block const& _block)
	{
		MemoryPrefix prefix;
		bool collecting = true;
		for (yul::Statement const& statement: _block.statements)
		{
			bool foundCall = false;
			if (collecting)
			{
				if (auto const* expression = std::get_if<yul::ExpressionStatement>(&statement))
					foundCall = scanExpressionForStaticCall(expression->expression, prefix);
				else if (auto const* assignment = std::get_if<yul::Assignment>(&statement))
					foundCall = assignment->value
						&& scanExpressionForStaticCall(*assignment->value, prefix);
				else if (auto const* declaration = std::get_if<yul::VariableDeclaration>(&statement))
					foundCall = declaration->value
						&& scanExpressionForStaticCall(*declaration->value, prefix);
			}

			bool const branch =
				std::holds_alternative<yul::If>(statement)
				|| std::holds_alternative<yul::ForLoop>(statement)
				|| std::holds_alternative<yul::Switch>(statement)
				|| std::holds_alternative<yul::Block>(statement);
			if (collecting && !foundCall && !branch)
				recordMemoryStore(statement, prefix);
			if (foundCall || branch)
				collecting = false;

			if (auto const* function = std::get_if<yul::FunctionDefinition>(&statement))
				collectStaticExternalCalls(function->body);
			else if (auto const* conditional = std::get_if<yul::If>(&statement))
				collectStaticExternalCalls(conditional->body);
			else if (auto const* loop = std::get_if<yul::ForLoop>(&statement))
			{
				collectStaticExternalCalls(loop->pre);
				collectStaticExternalCalls(loop->body);
				collectStaticExternalCalls(loop->post);
			}
			else if (auto const* switchStatement = std::get_if<yul::Switch>(&statement))
				for (yul::Case const& switchCase: switchStatement->cases)
					collectStaticExternalCalls(switchCase.body);
			else if (auto const* block = std::get_if<yul::Block>(&statement))
				collectStaticExternalCalls(*block);
		}
	}

	void operator()(yul::Identifier const& _identifier) override
	{
		recordExternalIdentifier(_identifier, ReadAccess);
	}

	void operator()(yul::Assignment const& _assignment) override
	{
		if (_assignment.value)
			visit(*_assignment.value);
		for (yul::Identifier const& identifier: _assignment.variableNames)
			recordExternalIdentifier(identifier, WriteAccess);
	}

	void operator()(yul::FunctionCall const& _call) override
	{
		std::string operation(yul::resolveFunctionName(_call.functionName, m_dialect));
		if (yul::BuiltinFunction const* builtin = yul::resolveBuiltinFunction(_call.functionName, m_dialect))
			classifyBuiltin(*builtin);
		else if (!m_localFunctions.count(operation))
			throw UnsupportedSolCore(
				"Inline assembly operation '" + operation
				+ "' is not a compiler-dialect builtin or a resolved local Yul function.");

		m_operations.insert(std::move(operation));
		yul::ASTWalker::operator()(_call);
	}

	void recordExternalIdentifier(yul::Identifier const& _identifier, unsigned _access)
	{
		auto reference = m_externalReferences.find(&_identifier);
		if (reference == m_externalReferences.end())
			return;
		m_seenExternalReferences.insert(&_identifier);

		auto const* variable
			= dynamic_cast<VariableDeclaration const*>(reference->second.declaration);
		if (!variable || !variable->annotation().type)
			throw UnsupportedSolCore(
				"Inline assembly external reference '" + _identifier.name.str()
				+ "' has no compiler-owned variable declaration and carrier type.");

		// Invariant: suffix admission is producer-typed. Solidity exposes
		// `.length` and `.offset` to inline assembly for dynamically-sized
		// calldata bytes/string references. Memory suffixes do not type-check in
		// solc, while ordinary calldata arrays remain outside this closed shape;
		// non-calldata `.offset` retains the existing storage-layout admission.
		auto const* arrayType = dynamic_cast<ArrayType const*>(variable->annotation().type);
		bool const isCalldataArray =
			arrayType && arrayType->location() == DataLocation::CallData;
		bool const isCalldataBytesString =
			isCalldataArray
			&& arrayType->isDynamicallySized()
			&& arrayType->isByteArrayOrString();
		bool const isCalldataBytesStringLength =
			reference->second.suffix == "length" && isCalldataBytesString;
		bool const isRepresentableOffset =
			reference->second.suffix == "offset"
			&& (!isCalldataArray || isCalldataBytesString);
		std::string suffix;
		if (reference->second.suffix.empty())
			suffix = "none";
		else if (
			reference->second.suffix == "slot"
			|| isRepresentableOffset
			|| isCalldataBytesStringLength)
			suffix = reference->second.suffix;
		else
			throw UnsupportedSolCore(
				"Inline assembly external reference '" + _identifier.name.str()
				+ "' uses unsupported suffix '." + reference->second.suffix
				+ "'; only none, .slot, non-calldata .offset, and calldata bytes/string .offset/.length are representable.");

		std::string declarationId = std::to_string(variable->id());
		auto declaration = m_declarations.find(declarationId);
		if (declaration == m_declarations.end())
		{
			Json carrierType = exportAssemblyCaptureType(*variable);
			m_declarations.emplace(
				declarationId,
				DeclarationInfo{variable, std::move(carrierType)});
		}
		else if (declaration->second.declaration != variable)
			throw UnsupportedSolCore(
				"Inline assembly external references contain a duplicate producer declaration id.");

		m_localAccesses[
			std::make_tuple(declarationId, variable->name(), std::move(suffix))] |= _access;
	}

	void classifyBuiltin(yul::BuiltinFunction const& _builtin)
	{
		std::string const& operation = _builtin.name;
		if (operation.rfind("verbatim_", 0) == 0)
			throw UnsupportedSolCore(
				"Inline assembly dialect operation '" + operation
				+ "' has opaque semantics that cannot be represented by the closed interface.");

		bool const isCall = inlineAssemblyOperationIs(
			operation,
			{"call", "callcode", "delegatecall", "staticcall", "create", "create2",
				"extcall", "extdelegatecall", "extstaticcall", "eofcreate"});
		bool const isLog
			= inlineAssemblyOperationIs(operation, {"log0", "log1", "log2", "log3", "log4"});
		bool const readsStorage = operation == "sload";
		bool const writesStorage = operation == "sstore";
		bool const readsTransient = operation == "tload";
		bool const writesTransient = operation == "tstore";
		bool const readsReturndata = operation == "returndatasize";
		bool const copiesReturndata = operation == "returndatacopy";
		bool const terminates = inlineAssemblyOperationIs(
			operation,
			{"stop", "return", "revert", "invalid", "selfdestruct", "returncontract"});
		bool const readsEnvironment = inlineAssemblyOperationIs(
			operation,
			{"address", "balance", "origin", "caller", "callvalue",
				"calldataload", "calldatasize", "calldatacopy",
				"codesize", "codecopy", "gasprice",
				"extcodesize", "extcodecopy", "extcodehash",
				"blockhash", "coinbase", "timestamp", "number",
				"difficulty", "prevrandao", "gaslimit", "chainid",
				"selfbalance", "basefee", "blobhash", "blobbasefee",
				"pc", "gas", "dataload", "dataloadn", "auxdataloadn",
				"datasize", "dataoffset", "datacopy"});

		if (isCall)
			m_effects.insert("call");
		if (isLog)
			m_effects.insert("log");
		if (readsStorage)
			m_effects.insert("storage_read");
		if (writesStorage)
			m_effects.insert("storage_write");
		if (readsTransient)
			m_effects.insert("transient_read");
		if (writesTransient)
			m_effects.insert("transient_write");
		if (readsReturndata)
			m_effects.insert("returndata_read");
		if (copiesReturndata)
			m_effects.insert("returndata_copy");
		if (terminates)
			m_effects.insert("termination");
		if (readsEnvironment)
			m_environmentReads.insert(operation);

		auto const& sideEffects = _builtin.sideEffects;
		if (sideEffects.memory == yul::SideEffects::Read)
			m_memoryReads = true;
		else if (sideEffects.memory == yul::SideEffects::Write)
		{
			bool const readsAndWritesMemory = inlineAssemblyOperationIs(
				operation, {"mcopy", "call", "callcode", "delegatecall", "staticcall"});
			bool const writesOnlyMemory = inlineAssemblyOperationIs(
				operation,
				{"mstore", "mstore8", "calldatacopy", "codecopy", "extcodecopy",
					"returndatacopy", "datacopy", "setimmutable"});
			if (!readsAndWritesMemory && !writesOnlyMemory)
				throw UnsupportedSolCore(
					"Inline assembly dialect operation '" + operation
					+ "' has memory semantics that cannot be represented exactly.");
			m_memoryReads = m_memoryReads || readsAndWritesMemory;
			m_memoryWrites = true;
		}

		bool const representsStorage = isCall || readsStorage || writesStorage;
		bool const representsTransient = isCall || readsTransient || writesTransient;
		bool const representsOtherState
			= isCall || isLog || terminates || readsEnvironment
			  || readsReturndata || copiesReturndata;
		if (
			sideEffects.storage != yul::SideEffects::None
			&& !representsStorage)
			throw UnsupportedSolCore(
				"Inline assembly dialect operation '" + operation
				+ "' has persistent-storage semantics outside the closed effect set.");
		if (
			sideEffects.transientStorage != yul::SideEffects::None
			&& !representsTransient)
			throw UnsupportedSolCore(
				"Inline assembly dialect operation '" + operation
				+ "' has transient-storage semantics outside the closed effect set.");
		if (
			sideEffects.otherState != yul::SideEffects::None
			&& !representsOtherState)
			throw UnsupportedSolCore(
				"Inline assembly dialect operation '" + operation
				+ "' has environment semantics outside the closed interface.");

		if (
			(readsStorage && sideEffects.storage != yul::SideEffects::Read)
			|| (writesStorage && sideEffects.storage != yul::SideEffects::Write)
			|| (readsTransient && sideEffects.transientStorage != yul::SideEffects::Read)
			|| (writesTransient && sideEffects.transientStorage != yul::SideEffects::Write))
			throw UnsupportedSolCore(
				"Inline assembly dialect metadata disagrees with the resolved operation '"
				+ operation + "'.");

		auto const& controlFlow = _builtin.controlFlowSideEffects;
		if (
			(controlFlow.canTerminate || controlFlow.canRevert || !controlFlow.canContinue)
			&& !terminates)
			throw UnsupportedSolCore(
				"Inline assembly dialect operation '" + operation
				+ "' has control-flow semantics outside the closed termination effect.");
	}

	yul::Dialect const& m_dialect;
	std::map<yul::Identifier const*, InlineAssemblyAnnotation::ExternalIdentifierInfo> const&
		m_externalReferences;
	std::set<std::string> m_localFunctions;
	std::set<yul::Identifier const*> m_seenExternalReferences;
	std::map<std::string, DeclarationInfo> m_declarations;
	std::map<std::tuple<std::string, std::string, std::string>, unsigned> m_localAccesses;
	bool m_memoryReads = false;
	bool m_memoryWrites = false;
	std::set<std::string> m_environmentReads;
	std::set<std::string> m_effects;
	std::set<std::string> m_operations;
	Json m_staticExternalCalls = Json::array();
};

Json exportInlineAssemblyInterface(InlineAssembly const& _assembly)
{
	return InlineAssemblyInterfaceCollector(_assembly).interfaceJson();
}

// --- Yul AST export functions ---

Json exportYulExpr(yul::Expression const& _expr, yul::Dialect const& _dialect);
Json exportYulStmt(yul::Statement const& _stmt, yul::Dialect const& _dialect);
Json exportYulBlock(yul::Block const& _block, yul::Dialect const& _dialect);

Json exportYulExpr(yul::Expression const& _expr, yul::Dialect const& _dialect)
{
	return std::visit(
		util::GenericVisitor{
			[&](yul::Literal const& _literal) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_literal";
				result["value"] = yul::formatLiteral(_literal);
				result["type"] = "u256";
				return result;
			},
			[&](yul::Identifier const& _identifier) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_local";
				result["name"] = _identifier.name.str();
				return result;
			},
			[&](yul::FunctionCall const& _call) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_call";
				result["function"] = std::string(yul::resolveFunctionName(_call.functionName, _dialect));
				result["args"] = Json::array();
				for (auto const& arg: _call.arguments)
					result["args"].emplace_back(exportYulExpr(arg, _dialect));
				return result;
			}},
		_expr);
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
	return std::visit(
		util::GenericVisitor{
			[&](yul::ExpressionStatement const& _exprStmt) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_expr";
				result["value"] = exportYulExpr(_exprStmt.expression, _dialect);
				return result;
			},
			[&](yul::Assignment const& _assignment) -> Json
			{
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
			[&](yul::VariableDeclaration const& _varDecl) -> Json
			{
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
			[&](yul::FunctionDefinition const& _funDef) -> Json
			{
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
			[&](yul::If const& _if) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_if";
				result["cond"] = exportYulExpr(*_if.condition, _dialect);
				result["body"] = exportYulBlock(_if.body, _dialect);
				return result;
			},
			[&](yul::Switch const& _switch) -> Json
			{
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
			[&](yul::ForLoop const& _for) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_for";
				result["pre"] = exportYulBlock(_for.pre, _dialect);
				result["cond"] = exportYulExpr(*_for.condition, _dialect);
				result["post"] = exportYulBlock(_for.post, _dialect);
				result["body"] = exportYulBlock(_for.body, _dialect);
				return result;
			},
			[&](yul::Break const&) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_break";
				return result;
			},
			[&](yul::Continue const&) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_continue";
				return result;
			},
			[&](yul::Leave const&) -> Json
			{
				Json result = Json::object();
				result["kind"] = "yul_leave";
				return result;
			},
			[&](yul::Block const& _block) -> Json { return exportYulBlock(_block, _dialect); }},
		_stmt);
}

// --- End Yul AST export functions ---
// Additive SolCore statement schema for the only assembly code-pointer cast
// admitted by the exporter:
//
//   {
//     "kind": "internal_fn_reinterpret",
//     "semantics": "identity_code_pointer",
//     "sourceDeclarationId": "<parameter AST id>",
//     "targetDeclarationId": "<return-parameter AST id>",
//     "sourceType": <internal_function>,
//     "targetType": <internal_function>,
//     "sourceTable": "<closed source table id>",
//     "targetTable": "<closed target table id>"
//   }
//
// Both declarations and both function types come from solc's typed AST.  The
// Yul is accepted only when it is exactly `output := input` between the sole
// input and sole return parameter of one function.  This preserves the code
// pointer while making the type/table transition explicit; no consumer has to
// inspect Yul or infer a cast from helper names.
std::optional<Json> exportInternalFnIdentityReinterpret(InlineAssembly const& _assembly)
{
	yul::Block const& root = _assembly.operations().root();
	if (root.statements.size() != 1)
		return std::nullopt;
	auto const* assignment = std::get_if<yul::Assignment>(&root.statements.front());
	if (!assignment || assignment->variableNames.size() != 1 || !assignment->value)
		return std::nullopt;
	auto const* rhs = std::get_if<yul::Identifier>(assignment->value.get());
	if (!rhs)
		return std::nullopt;

	auto const& references = _assembly.annotation().externalReferences;
	if (references.size() != 2)
		return std::nullopt;
	auto lhsReference = references.find(&assignment->variableNames.front());
	auto rhsReference = references.find(rhs);
	if (lhsReference == references.end() || rhsReference == references.end()
		|| !lhsReference->second.suffix.empty() || !rhsReference->second.suffix.empty())
		return std::nullopt;
	auto const* targetDeclaration
		= dynamic_cast<VariableDeclaration const*>(lhsReference->second.declaration);
	auto const* sourceDeclaration
		= dynamic_cast<VariableDeclaration const*>(rhsReference->second.declaration);
	if (!sourceDeclaration || !targetDeclaration || sourceDeclaration == targetDeclaration
		|| sourceDeclaration->scope() != targetDeclaration->scope()
		|| !sourceDeclaration->isCallableOrCatchParameter() || sourceDeclaration->isReturnParameter()
		|| !targetDeclaration->isReturnParameter())
		return std::nullopt;
	auto const* function = dynamic_cast<FunctionDefinition const*>(sourceDeclaration->scope());
	if (!function || function->parameters().size() != 1 || function->returnParameters().size() != 1
		|| function->parameters().front().get() != sourceDeclaration
		|| function->returnParameters().front().get() != targetDeclaration)
		return std::nullopt;

	auto const* sourceType = dynamic_cast<FunctionType const*>(sourceDeclaration->annotation().type);
	auto const* targetType = dynamic_cast<FunctionType const*>(targetDeclaration->annotation().type);
	if (!sourceType || !targetType || sourceType->kind() != FunctionType::Kind::Internal
		|| targetType->kind() != FunctionType::Kind::Internal)
		return std::nullopt;

	Json sourceWireType = exportResolvedType(sourceType);
	Json targetWireType = exportResolvedType(targetType);
	InternalFnTableRecord const& sourceTable = registerInternalFnTable(*sourceType, sourceWireType);
	InternalFnTableRecord const& targetTable = registerInternalFnTable(*targetType, targetWireType);
	Json result = Json::object();
	result["kind"] = "internal_fn_reinterpret";
	result["semantics"] = "identity_code_pointer";
	result["sourceDeclarationId"] = std::to_string(sourceDeclaration->id());
	result["targetDeclarationId"] = std::to_string(targetDeclaration->id());
	result["sourceType"] = std::move(sourceWireType);
	result["targetType"] = std::move(targetWireType);
	result["sourceTable"] = sourceTable.tableId;
	result["targetTable"] = targetTable.tableId;
	return result;
}


Json exportStmtDispatch(Statement const& _stmt)
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
		{
			if (activeResidualStorageRefReturn)
			{
				auto reference = exportStorageRefValue(*returnStmt->expression());
				if (!reference)
					throw UnsupportedSolCore(
						"Residual storage-reference return did not resolve to an exact typed path.");
				result["value"] = std::move(*reference);
			}
			else
				result["value"] = exportExpr(*returnStmt->expression());
		}
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

		// --- General storage-reference-variable alias tracking (Phase 1a §3.2) ---
		if (varDecl->declarations().size() == 1 && varDecl->declarations().front() && varDecl->initialValue())
		{
			VariableDeclaration const& decl = *varDecl->declarations().front();
			if (decl.referenceLocation() == VariableDeclaration::Location::Storage
				&& !storageRefNeverTrack.count(&decl))
			{
				if (auto resolved = resolveStorageRefInitializer(*varDecl->initialValue(), decl))
				{
					// Keep the existing fail-closed gate until dynamic-array
					// delete is lowered through StorageArray.clear rather than
					// replacing the backing sparse slot map.
					if (storageRefTargetRacesArrayShrink(resolved->first))
						throw UnsupportedSolCore(
							"Storage reference bound to an element of `" + resolved->first.root
							+ "`, which this function can shrink (`pop()`/`delete`/whole "
							  "reassignment), is not modeled until every shrink path preserves "
							  "the bound slot identity.");
					storageRefAliasTargets[&decl] = resolved->first;
					Json result = Json::object();
					result["kind"] = "block";
					result["statements"] = Json::array();
					for (Json const& snapshotLet: resolved->second)
						result["statements"].emplace_back(snapshotLet);
					return result;
				}
				// Unresolved: fall through to the status-quo copy-`let` lowering
				// below (§3.6 fallback policy — zero-regression guarantee).
			}
		}
		if (varDecl->declarations().size() == 1 && varDecl->declarations().front() && varDecl->initialValue())
		{
			VariableDeclaration const& declaration = *varDecl->declarations().front();
			if (declaration.referenceLocation() == VariableDeclaration::Location::Storage
				&& (storageRefNeverTrack.count(&declaration) || storageRefWriteThroughLocals.count(&declaration)))
				if (auto reference = exportStorageRefValue(*varDecl->initialValue()))
				{
					storageRefValueLocals.insert(&declaration);
					Json result = Json::object();
					result["kind"] = "let";
					result["sourceDeclarationId"] = std::to_string(declaration.id());
					result["name"] = declaration.name();
					result["type"] = storageRefWireType(declaration.annotation().type);
					result["value"] = std::move(*reference);
					return result;
				}
		}
		// --- End general storage-reference-variable alias tracking ---

		// [P0 fail-closed: E0(a')] Every storage-located declaration that
		// reaches this point is UNTRACKED — the namespaced-alias arm and the
		// resolved-alias arm above both `return` on success, so control only
		// gets here when the initializer did not resolve to a storage path
		// (call-returned reference), when the local is never-tracked (rebound
		// somewhere in the function), or when there is no initializer at all
		// (declare-then-assign). The copy-`let` lowering below binds a VALUE
		// copy; a later write through this local would be exported as a rebind
		// of that dead copy and silently dropped. Refuse instead.
		//
		// Reads-only copies stay accepted: they are value-correct at the
		// binding point, and the scanner records only genuine write-throughs.
		for (auto const& declaredVar: varDecl->declarations())
			if (declaredVar && declaredVar->referenceLocation() == VariableDeclaration::Location::Storage)
				// [E2(b)] A single-assignment (declare-then-assign) local's
				// binding decision — track, value-copy fallback, or the
				// E0(a') refusal — is owned by its assignment statement (see
				// exportAssignment); the guard must not fire before the bind
				// site has been reached.
				if (!storageRefSingleAssignBindable.count(declaredVar.get()))
					requireNoWriteThroughUntrackedStorageLocalOrThrow(*declaredVar);

		// Single variable declaration with initializer (common case)
		if (varDecl->declarations().size() == 1 && varDecl->declarations().front() && varDecl->initialValue())
		{
			auto const& decl = *varDecl->declarations().front();
			Json result = Json::object();
			result["kind"] = "let";
			result["sourceDeclarationId"] = std::to_string(decl.id());
			result["name"] = decl.name();
			Json location = dataLocationEntry(decl);
			bool storage = location.is_string() && location.get<std::string>() == "storage";
			if (!location.is_null())
				result["location"] = location;
			try
			{
				result["type"] = exportTypeName(decl.typeName(), storage);
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
			auto const& decl = *varDecl->declarations().front();
			Json result = Json::object();
			result["kind"] = "let";
			result["sourceDeclarationId"] = std::to_string(decl.id());
			result["name"] = decl.name();
			Json location = dataLocationEntry(decl);
			bool storage = location.is_string() && location.get<std::string>() == "storage";
			if (!location.is_null())
				result["location"] = location;
			try
			{
				result["type"] = exportTypeName(decl.typeName(), storage);
				result["value"] = defaultValueForTypeName(decl.typeName(), storage);
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
				letStmt["sourceDeclarationId"] = std::to_string(decl->id());
				letStmt["name"] = decl->name();
				Json location = dataLocationEntry(*decl);
				bool storage = location.is_string() && location.get<std::string>() == "storage";
				if (!location.is_null())
					letStmt["location"] = location;
				try
				{
					letStmt["type"] = exportTypeName(decl->typeName(), storage);
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
						letStmt["value"] = defaultValueForTypeName(decl->typeName(), storage);
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
			return exportAssignment(
				assignment->leftHandSide(), assignment->assignmentOperator(), assignment->rightHandSide());

		if (auto const* unary = dynamic_cast<UnaryOperation const*>(&expr))
		{
			if (unary->getOperator() == Token::Inc || unary->getOperator() == Token::Dec)
				return exportUnaryMutation(unary->subExpression(), unary->getOperator());
			if (unary->getOperator() == Token::Delete)
				return exportDelete(unary->subExpression());
		}

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
					Json payload = exportRevertPayload(*call, /*_unconditionalPayload=*/false);
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
					Json payload = exportRevertPayload(*call, /*_unconditionalPayload=*/true);
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
				// Delegates to lowerInternalCalleeAndArgs — see the comment on
				// the expression-position case in exportExpr.
				auto const* funcDef
					= dynamic_cast<FunctionDefinition const*>(callee->annotation().referencedDeclaration);
				if (funcDef)
				{
					std::string plainName = virtualCallTargetName(*funcDef);
					FunctionDefinition const& resolvedImpl = resolveInternalCallImplementation(*funcDef);
					auto [calleeName, callArgs] = lowerInternalCalleeAndArgs(*call, resolvedImpl, plainName);
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = std::move(calleeName);
					callExpr["args"] = std::move(callArgs);
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
					std::string plainName = flattenedStaticBaseCallName(*resolved);
					auto [calleeName, callArgs] = lowerInternalCalleeAndArgs(*call, *resolved->target, plainName);
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = std::move(calleeName);
					addStaticBaseCallTarget(callExpr, *resolved);
					callExpr["args"] = std::move(callArgs);
					result["value"] = callExpr;
					return result;
				}

				if (memberAccess->memberName() == "push" || memberAccess->memberName() == "pop")
				{
					// §2c case 1: intercept BUILTIN push/pop only, gated on
					// the callee's FunctionType::Kind. A using-for `push`/
					// `pop` (e.g. `using MemArrLib for uint256[]` on a MEMORY
					// array, where no builtin push exists) resolves to a
					// library FunctionDefinition and must fall through to the
					// using-for internal-call branch below — intercepting the
					// member NAME for any array receiver reproduced exactly
					// the upstream solidity-lean bug this corpus pins.
					auto const* pushPopFnType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type);
					bool builtinPushPop = pushPopFnType
										  && (pushPopFnType->kind() == FunctionType::Kind::ArrayPush
											  || pushPopFnType->kind() == FunctionType::Kind::ArrayPop);
					Type const* baseType = memberAccess->expression().annotation().type;
					if (builtinPushPop && baseType && baseType->category() == Type::Category::Array)
					{
						if (hasResidualStorageRefRoot(memberAccess->expression()))
						{
							auto reference = exportStorageRefValue(memberAccess->expression());
							if (!reference)
								throw UnsupportedSolCore("Residual storage-array push/pop base did not resolve.");
							std::string tempName
								= "__solcore_residual_array_sref_" + std::to_string(stableSyntheticNodeId(*call));
							Json letReference = Json::object();
							letReference["kind"] = "let";
							letReference["sourceDeclarationId"] = Json();
							letReference["name"] = tempName;
							letReference["type"] = storageRefWireType(baseType);
							letReference["value"] = std::move(*reference);

							Json update = Json::object();
							update["kind"] = "internal_call";
							if (memberAccess->memberName() == "pop")
								update["function"] = "array_pop_expr";
							else if (call->arguments().empty())
								update["function"] = "array_push_default_expr";
							else
								update["function"] = "array_push_expr";
							update["args"] = Json::array();
							update["args"].emplace_back(storageRefGetJson(localExpr(tempName), baseType));
							if (memberAccess->memberName() == "push" && !call->arguments().empty())
								update["args"].emplace_back(exportExpr(*call->arguments().front()));

							Json set = Json::object();
							set["kind"] = "storage_ref_set";
							set["referentType"] = exportResolvedType(baseType, true);
							set["reference"] = localExpr(tempName);
							set["value"] = std::move(update);

							Json block = Json::object();
							block["kind"] = "block";
							block["statements"] = Json::array();
							block["statements"].emplace_back(std::move(letReference));
							block["statements"].emplace_back(std::move(set));
							return block;
						}

						// A whole state-array identifier keeps the compact native
						// ArrayPush/ArrayPop statement. Every other accepted base
						// must resolve structurally to an exact state root or
						// copy-in/copy-out storage parameter. Rooted field,
						// mapping, index, tracked-alias, and static call-return
						// paths then use the typed functional array operation;
						// Base.ml reconstructs the exact update spine and writes
						// it back to that root.
						bool baseIsPlainStateVarIdent = false;
						if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
						{
							auto const* decl = dynamic_cast<VariableDeclaration const*>(
								baseIdent->annotation().referencedDeclaration);
							baseIsPlainStateVarIdent = decl && decl->isStateVariable();
						}
						if (!baseIsPlainStateVarIdent)
							if (auto rootedBase = exportResolvedStorageRefUse(
									memberAccess->expression(), StorageRefKeySnapshotMode::OrderedPath))
							{
								bool const hasPushedValue
									= memberAccess->memberName() == "push" && !call->arguments().empty();
								Json pushedValue;
								if (hasPushedValue)
								{
									Expression const& argument = *call->arguments().front();
									pushedValue = evalOrderRelevant(argument)
													  ? pinExpressionOnce(argument, "array_push_value")
													  : exportExpr(argument);
								}
								Json result = Json::object();
								result["kind"] = "expr";
								Json callExpr = Json::object();
								callExpr["kind"] = "internal_call";
								if (memberAccess->memberName() == "pop")
									callExpr["function"] = "array_pop_expr";
								else if (call->arguments().empty())
									callExpr["function"] = "array_push_default_expr";
								else
									callExpr["function"] = "array_push_expr";
								callExpr["args"] = Json::array();
								callExpr["args"].emplace_back(std::move(*rootedBase));
								if (hasPushedValue)
									callExpr["args"].emplace_back(std::move(pushedValue));
								result["value"] = std::move(callExpr);
								return result;
							}

						// Extract base_path as a string list for the OCaml parser
						Json basePath = Json::array();
						if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
						{
							auto const* decl = dynamic_cast<VariableDeclaration const*>(
								baseIdent->annotation().referencedDeclaration);
							if (decl && decl->isStateVariable())
								basePath.emplace_back(decl->name());
							else if (decl)
								throw UnsupportedSolCore("Array push/pop on local storage reference is unsupported.");
							else
								basePath.emplace_back(baseIdent->name());
						}
						else
							throw UnsupportedSolCore(
								"Array push/pop statement with complex base expression is unsupported.");

						if (memberAccess->memberName() == "push")
						{
							Json result = Json::object();
							if (call->arguments().empty())
								result["kind"] = "array_push_default";
							else
							{
								result["kind"] = "array_push";
								result["value"] = exportExpr(*call->arguments().front());
							}
							result["base_path"] = basePath;
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

				// Same typed attached-call lowering as expression position;
				// wrapping it as an expression statement must not lose the
				// receiver or exact library identity.
				if (auto usingForCall = exportUsingForCall(*call, *memberAccess))
				{
					Json result = Json::object();
					result["kind"] = "expr";
					result["value"] = *usingForCall;
					return result;
				}

				// External contract call as statement: contract.method(args)
				Type const* baseType = memberAccess->expression().annotation().type;
				if (baseType && baseType->category() == Type::Category::Contract)
				{
					if (auto externalCall = exportExternalContractCall(*call, *memberAccess, true))
						return *externalCall;
				}

				// Library-qualified or type-qualified function call as statement: L.f(args)
				if (auto const* funcDef
					= dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
				{
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					std::string plainName = isPublicLibraryStructuralStorageFunction(*funcDef)
												? storageRefInternalEntryName(*funcDef)
												: exportedFunctionName(*funcDef);
					auto [calleeName, callArgs] = lowerInternalCalleeAndArgs(*call, *funcDef, std::move(plainName));
					callExpr["function"] = std::move(calleeName);
					addInternalLibraryCallContractId(callExpr, *funcDef);
					callExpr["args"] = std::move(callArgs);
					result["value"] = std::move(callExpr);
					return result;
				}
			}

			// Options-aware low-level call as a statement. Staticcall admits
			// only gas and reuses the existing gas-abstracted raw-call result.
			if (auto const* options = dynamic_cast<FunctionCallOptions const*>(&call->expression()))
			{
				if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&options->expression()))
				{
					if (auto externalCall = exportExternalContractCall(*call, *memberAccess, true))
						return *externalCall;

					if (isOptionsAwareLowLevelCall(memberAccess->memberName()))
					{
						validateLowLevelCallOptions(memberAccess->memberName(), *options);
						CallEvaluation evaluation = exportCallEvaluation(*call, memberAccess->expression(), options);
						Json result = Json::object();
						result["kind"] = "expr";
						Json callExpr = Json::object();
						callExpr["kind"] = "low_level_call";
						callExpr["callKind"] = lowLevelCallKindString(memberAccess->memberName());
						callExpr["target"] = std::move(evaluation.target);

						Json valueExpr = u256Literal("0");
						for (size_t i = 0; i < options->names().size(); ++i)
							if (*options->names()[i] == "value")
								valueExpr = std::move(evaluation.options[i]);
						callExpr["value"] = std::move(valueExpr);

						if (!evaluation.arguments.empty())
							callExpr["data"] = std::move(evaluation.arguments.front());
						else
						{
							callExpr["data"] = emptyBytesLiteral();
						}
						result["value"] = std::move(callExpr);
						return result;
					}
				}
			}

			// Low-level call as statement without options:
			// address.call(data), address.delegatecall(data), or address.staticcall(data).
			if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
			{
				if (isOptionsAwareLowLevelCall(memberAccess->memberName()))
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
							callExpr["data"] = emptyBytesLiteral();
						}
						result["value"] = callExpr;
						return result;
					}
				}
			}

			// address.send(x) as a statement (result discarded): mirror the
			// low-level-call statement lowering above exactly — same node
			// shape the generator's statement-level state threading already
			// handles for `addr.call{value:x}("");` — instead of routing
			// through the expression path's tuple_get projection (which would
			// put a stateful call in expression position). §2b; the
			// expression path handles value-position `.send` and the
			// fail-closed `.transfer` refusal.
			if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
			{
				auto const* calleeFnType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type);
				if (calleeFnType && calleeFnType->kind() == FunctionType::Kind::Send)
				{
					if (call->arguments().size() != 1)
						throw UnsupportedSolCore("address.send expects exactly one value argument.");
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "low_level_call";
					callExpr["callKind"] = lowLevelCallKindString("call");
					callExpr["target"] = exportExpr(memberAccess->expression());
					callExpr["value"] = exportExpr(*call->arguments().front());
					callExpr["data"] = emptyBytesLiteral();
					result["value"] = callExpr;
					return result;
				}
				// `a.transfer(x);` — §2b follow-up (dapphub-weth9): transfer
				// IS send + revert-on-failure, so lower it to the exact
				// statement pair the pipeline already handles green for
				// `(bool ok, ) = a.call{value: x}(""); require(ok);`:
				//   let <tmp> : bool = tuple_get(low_level_call(...), 0)
				//   require(<tmp>)
				// The callee over-approximation argument is identical to
				// `.send` (the modeled arbitrary callee can do strictly more
				// than the real 2300-gas-stipend callee; we never prove
				// anything FROM the stipend). The failure branch reverts —
				// matching real transfer-failure control flow; transfer
				// returns no value, so statement position is the only
				// reachable shape (the expression-path refusal stays as
				// defense).
				if (calleeFnType && calleeFnType->kind() == FunctionType::Kind::Transfer)
				{
					if (call->arguments().size() != 1)
						throw UnsupportedSolCore("address.transfer expects exactly one value argument.");
					Json lowLevel = Json::object();
					lowLevel["kind"] = "low_level_call";
					lowLevel["callKind"] = lowLevelCallKindString("call");
					lowLevel["target"] = exportExpr(memberAccess->expression());
					lowLevel["value"] = exportExpr(*call->arguments().front());
					lowLevel["data"] = emptyBytesLiteral();

					Json okValue = Json::object();
					okValue["kind"] = "internal_call";
					okValue["function"] = "tuple_get";
					okValue["args"] = Json::array();
					okValue["args"].emplace_back(std::move(lowLevel));
					Json index = Json::object();
					index["kind"] = "u256";
					index["value"] = "0";
					okValue["args"].emplace_back(std::move(index));

					std::string tempName = "__solcore_transfer_ok_" + std::to_string(stableSyntheticNodeId(*call));
					Json letStmt = Json::object();
					letStmt["kind"] = "let";
					letStmt["sourceDeclarationId"] = Json();
					letStmt["name"] = tempName;
					letStmt["type"] = "bool";
					letStmt["value"] = std::move(okValue);

					Json requireStmt = Json::object();
					requireStmt["kind"] = "require";
					Json condLocal = Json::object();
					condLocal["kind"] = "local";
					condLocal["name"] = tempName;
					requireStmt["cond"] = std::move(condLocal);

					Json block = Json::object();
					block["kind"] = "block";
					block["statements"] = Json::array();
					block["statements"].emplace_back(std::move(letStmt));
					block["statements"].emplace_back(std::move(requireStmt));
					return block;
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

		// Extract the event name from the call expression. Prefer the
		// EventDefinition solc itself resolved, so the emitted name comes from
		// the same pre-assigned memo the declaration used — an overloaded
		// `emit E(a)` then binds to the same `E_address` its declaration was
		// exported under, by construction rather than by string coincidence.
		Declaration const* eventDecl = nullptr;
		std::string rawEventName;
		if (auto const* callee = dynamic_cast<Identifier const*>(&call.expression()))
		{
			eventDecl = callee->annotation().referencedDeclaration;
			rawEventName = callee->name();
		}
		else if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call.expression()))
		{
			eventDecl = memberAccess->annotation().referencedDeclaration;
			rawEventName = memberAccess->memberName();
		}
		else
			throw UnsupportedSolCore("Unsupported emit expression in SolCore exporter.");

		auto const* eventDef = dynamic_cast<EventDefinition const*>(eventDecl);
		if (!eventDef)
			throw UnsupportedSolCore(
				"emit of event '" + rawEventName
				+ "' could not be resolved to its declaration-owned parameter types.");
		result["event"] = exportedEventName(*eventDef);
		result["eventDeclarationId"] = std::to_string(static_cast<int64_t>(eventDef->id()));

		auto const& parameters = eventDef->parameters();
		if (call.arguments().size() != parameters.size())
			throw UnsupportedSolCore(
				"emit argument count does not match its compiler-resolved event declaration.");

		bool observable = false;
		bool hasIndexed = false;
		for (size_t i = 0; i < call.arguments().size(); ++i)
		{
			observable = observable || evalOrderRelevant(*call.arguments()[i]);
			hasIndexed = hasIndexed || parameters[i]->isIndexed();
		}

		result["args"] = Json::array();
		if (!observable)
			for (auto const& arg: call.arguments())
				result["args"].emplace_back(exportExpr(*arg));
		else
		{
			std::vector<Json> evaluated(call.arguments().size());
			auto captureArgument = [&](size_t index)
			{
				evaluated[index] = captureMeasuredOrderChild(
					*call.arguments()[index], "emit_arg_" + std::to_string(index));
			};

			if (hasIndexed)
			{
				if (!activeCompilerStack)
					throw UnsupportedSolCore(
						"indexed emit arguments with observable evaluation order "
						"require an explicit legacy codegen identity.");
				if (activeCompilerStack->viaIR())
					throw UnsupportedSolCore(
						"indexed emit arguments with observable evaluation order "
						"diverge under via-IR; only the measured legacy indexed-first "
						"row is admitted.");

				// ExpressionCompiler's legacy Event arm evaluates indexed
				// arguments in reverse source order, then data arguments
				// forward. Reconstruction below remains in declaration order.
				for (size_t i = call.arguments().size(); i > 0; --i)
					if (parameters[i - 1]->isIndexed())
						captureArgument(i - 1);
				for (size_t i = 0; i < call.arguments().size(); ++i)
					if (!parameters[i]->isIndexed())
						captureArgument(i);
			}
			else
				for (size_t i = 0; i < call.arguments().size(); ++i)
					captureArgument(i);

			for (Json& argument: evaluated)
				result["args"].emplace_back(std::move(argument));
		}
		result["argTypes"] = Json::array();
		for (auto const& parameter: parameters)
		{
			if (!parameter->annotation().type)
				throw UnsupportedSolCore(
					"Emit declaration parameter has no compiler-resolved type.");
			result["argTypes"].emplace_back(exportResolvedType(parameter->annotation().type));
		}
		return result;
	}

	if (auto const* revertStmt = dynamic_cast<RevertStatement const*>(&_stmt))
	{
		Json result = Json::object();
		result["kind"] = "revert";
		Json payload = exportRevertPayload(revertStmt->errorCall(), /*_unconditionalPayload=*/true);
		for (auto const& [key, value]: payload.items())
			result[key] = value;
		return result;
	}

	if (auto const* asmStmt = dynamic_cast<InlineAssembly const*>(&_stmt))
	{
		// Yul can manufacture or mutate a function code pointer only if the
		// assembly can reach a Solidity value that carries one. The one
		// exception is a structurally verified identity reinterpretation:
		// export it as a typed table transition instead of opaque Yul.
		bool touchesInternalFnValue = false;
		for (auto const& entry: asmStmt->annotation().externalReferences)
			if (auto const* declaration = dynamic_cast<VariableDeclaration const*>(entry.second.declaration))
				if (typeContainsInternalFnValue(declaration->annotation().type))
					touchesInternalFnValue = true;
		if (touchesInternalFnValue)
		{
			if (auto reinterpretation = exportInternalFnIdentityReinterpret(*asmStmt))
				return std::move(*reinterpretation);
			// Any other variant remains opaque and poisons every closed table
			// that depends on it during finalization below.  This preserves the
			// existing fail-closed behavior for arbitrary code-pointer writes.
			internalFnAssemblyTouchesValue = true;
		}

		internalFnContractContainsAssembly = true;
		Json result = Json::object();
		result["kind"] = "inline_assembly";
		result["body"] = exportYulBlock(asmStmt->operations().root(), asmStmt->dialect());
		result["interface"] = exportInlineAssemblyInterface(*asmStmt);
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
		// A do-while whose condition embeds exactly one PREFIX ++/-- of a
		// plain stack local desugars structurally: the condition evaluates
		// once per iteration AFTER the body, so appending the mutation as a
		// statement and comparing the plain local afterwards is
		// observation-equivalent for every legacy evaluation order PROVIDED
		// the other operand is side-effect-free and independent of the
		// mutated local (witness family: solady `do { ... } while (--i != 0)`
		// / `while (--i != c)`; FixedPointMathLib::lambertW0Wad). Checked or
		// unchecked semantics ride on exportUnaryMutation, the same
		// statement-position lowering `--i;` uses. Every other mutating
		// condition falls through to the established fail-closed refusal in
		// requireHoistScopeAllowed.
		if (whileStmt->isDoWhile())
			if (auto const* cond = dynamic_cast<BinaryOperation const*>(&whileStmt->condition()))
			{
				auto comparisonKind = [&](Token _op, Type const* _commonType) -> char const* {
					auto const* integerType = dynamic_cast<IntegerType const*>(_commonType);
					bool const signedType = integerType && integerType->isSigned();
					switch (_op)
					{
					case Token::Equal: return "u256_eq";
					case Token::NotEqual: return "u256_ne";
					case Token::LessThan: return signedType ? "i256_lt" : "u256_lt";
					case Token::LessThanOrEqual: return signedType ? "i256_le" : "u256_le";
					case Token::GreaterThan: return signedType ? "i256_gt" : "u256_gt";
					case Token::GreaterThanOrEqual: return signedType ? "i256_ge" : "u256_ge";
					default: return nullptr;
					}
				};
				auto prefixMutation = [](Expression const& _operand) -> UnaryOperation const* {
					auto const* unary = dynamic_cast<UnaryOperation const*>(&_operand);
					if (unary && unary->isPrefixOperation()
						&& (unary->getOperator() == Token::Inc || unary->getOperator() == Token::Dec))
						return unary;
					return nullptr;
				};
				UnaryOperation const* mutation = prefixMutation(cond->leftExpression());
				Expression const* other = mutation ? &cond->rightExpression() : nullptr;
				if (!mutation)
				{
					mutation = prefixMutation(cond->rightExpression());
					other = mutation ? &cond->leftExpression() : nullptr;
				}
				Type const* commonType = cond->annotation().commonType;
				char const* kind = comparisonKind(cond->getOperator(), commonType);
				auto const* mutatedIdent = mutation
					? dynamic_cast<Identifier const*>(&mutation->subExpression())
					: nullptr;
				auto const* mutatedDecl = mutatedIdent
					? dynamic_cast<VariableDeclaration const*>(mutatedIdent->annotation().referencedDeclaration)
					: nullptr;
				bool const plainStackLocal = mutatedDecl && !mutatedDecl->isStateVariable()
					&& mutatedDecl->referenceLocation() != VariableDeclaration::Location::Storage
					&& namespacedStorageAliases.count(mutatedDecl->name()) == 0;
				// The independent operand may be side-effect-free directly or a
				// pure elementary-type conversion of one (the witness family
				// compares against `uint256(0)`); the cast neither reads nor
				// writes anything its argument does not.
				auto sideEffectFreeOrPureCast = [](Expression const& _expr) -> Expression const* {
					if (isSideEffectFreeExpr(_expr))
						return &_expr;
					if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
						if (*call->annotation().kind == FunctionCallKind::TypeConversion
							&& call->arguments().size() == 1
							&& isSideEffectFreeExpr(*call->arguments().front()))
							return call->arguments().front().get();
					return nullptr;
				};
				Expression const* otherCore = other ? sideEffectFreeOrPureCast(*other) : nullptr;
				bool otherIndependent = otherCore != nullptr;
				if (otherIndependent)
				{
					struct DeclRefScanner: ASTConstVisitor
					{
						VariableDeclaration const* decl;
						bool found = false;
						explicit DeclRefScanner(VariableDeclaration const* _decl): decl(_decl) {}
						bool visit(Identifier const& _ident) override
						{
							if (_ident.annotation().referencedDeclaration == decl)
								found = true;
							return !found;
						}
					} scanner{mutatedDecl};
					otherCore->accept(scanner);
					otherIndependent = !scanner.found;
				}
				bool const integerFamily = commonType
					&& (dynamic_cast<IntegerType const*>(commonType) || dynamic_cast<RationalNumberType const*>(commonType));
				// `continue` transfers to the CONDITION, whose embedded
				// mutation still runs on the EVM; the desugared form places
				// the mutation at the body tail, which `continue` would skip.
				// Refuse the transform when the body contains a continue
				// bound to THIS loop (inner loops own their own continues);
				// `break` needs no guard — it exits without evaluating the
				// condition, exactly like the desugared tail-skip.
				struct LoopContinueScanner: ASTConstVisitor
				{
					bool found = false;
					unsigned depth = 0;
					bool visit(WhileStatement const&) override { ++depth; return true; }
					void endVisit(WhileStatement const&) override { --depth; }
					bool visit(ForStatement const&) override { ++depth; return true; }
					void endVisit(ForStatement const&) override { --depth; }
					bool visit(Continue const&) override
					{
						if (depth == 0)
							found = true;
						return false;
					}
				} continueScanner;
				whileStmt->body().accept(continueScanner);
				bool const noOwnContinue = !continueScanner.found;
				if (mutation && kind && plainStackLocal && otherIndependent && integerFamily
					&& noOwnContinue)
				{
					Json condition = Json::object();
					condition["kind"] = kind;
					Json mutatedRead = exportExpr(mutation->subExpression());
					Json otherValue = exportExpr(*other);
					condition["lhs"] = (&cond->leftExpression() == static_cast<Expression const*>(other))
						? std::move(otherValue) : std::move(mutatedRead);
					condition["rhs"] = (&cond->leftExpression() == static_cast<Expression const*>(other))
						? exportExpr(mutation->subExpression()) : exportExpr(*other);
					Json bodyBlock = Json::object();
					bodyBlock["kind"] = "block";
					bodyBlock["statements"] = Json::array();
					bodyBlock["statements"].emplace_back(exportStmt(whileStmt->body()));
					bodyBlock["statements"].emplace_back(
						exportUnaryMutation(mutation->subExpression(), mutation->getOperator()));
					result["kind"] = "do_while";
					result["cond"] = std::move(condition);
					result["body"] = std::move(bodyBlock);
					return result;
				}
			}
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
		for (auto const& clause: tryStmt->clauses())
		{
			Json c = Json::object();
			c["error_name"] = clause->errorName().empty() ? Json() : Json(clause->errorName());

			if (clause->parameters())
			{
				c["params"] = Json::array();
				for (auto const& param: clause->parameters()->parameters())
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

/// The statement's once-evaluated header expressions: the expression
/// subtrees this statement evaluates exactly once, unconditionally, before
/// (or as) its own effect. These are the positions from which a mutating
/// sub-expression may be hoisted to just before the statement, and they
/// form the conflict-scan domain (see HoistScope). Statement kinds whose
/// directly-embedded expressions are re-evaluated (while/for conditions)
/// or that have none return an empty list — hoisting is then disallowed
/// under them (nested statements install their own scopes).
std::vector<Expression const*> statementHeaderExpressions(Statement const& _stmt)
{
	if (auto const* exprStmt = dynamic_cast<ExpressionStatement const*>(&_stmt))
		return {&exprStmt->expression()};
	if (auto const* ifStmt = dynamic_cast<IfStatement const*>(&_stmt))
		// Only the condition: the branches are their own statements.
		return {&ifStmt->condition()};
	if (auto const* returnStmt = dynamic_cast<Return const*>(&_stmt))
	{
		if (returnStmt->expression())
			return {returnStmt->expression()};
		return {};
	}
	if (auto const* varDeclStmt = dynamic_cast<VariableDeclarationStatement const*>(&_stmt))
	{
		if (varDeclStmt->initialValue())
			return {varDeclStmt->initialValue()};
		return {};
	}
	if (auto const* emitStmt = dynamic_cast<EmitStatement const*>(&_stmt))
		return {&emitStmt->eventCall()};
	if (auto const* revertStmt = dynamic_cast<RevertStatement const*>(&_stmt))
		return {&revertStmt->errorCall()};
	if (auto const* tryStmt = dynamic_cast<TryStatement const*>(&_stmt))
		return {&tryStmt->externalCall()};
	// Block, While (condition re-evaluated each iteration!), For (ditto),
	// Break, Continue, InlineAssembly, PlaceholderStatement, ...
	return {};
}

Json exportStmt(Statement const& _stmt)
{
	HoistScopeGuard hoistScope(statementHeaderExpressions(_stmt));
	Json result = exportStmtDispatch(_stmt);
	return hoistScope.wrap(std::move(result));
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
FunctionDefinition const*
resolveWriteOracleCallTarget(Expression const& _callee, ContractDefinition const* _mostDerivedContract)
{
	Declaration const* referenced = nullptr;
	bool requiresVirtual = false;
	if (auto const* ident = dynamic_cast<Identifier const*>(&_callee))
	{
		referenced = ident->annotation().referencedDeclaration;
		requiresVirtual
			= ident->annotation().requiredLookup.set() && *ident->annotation().requiredLookup == VirtualLookup::Virtual;
	}
	else if (auto const* member = dynamic_cast<MemberAccess const*>(&_callee))
	{
		referenced = member->annotation().referencedDeclaration;
		requiresVirtual = member->annotation().requiredLookup.set()
						  && *member->annotation().requiredLookup == VirtualLookup::Virtual;
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
	return std::visit(
		util::GenericVisitor{
			[&](yul::Literal const&) -> bool { return false; },
			[&](yul::Identifier const&) -> bool { return false; },
			[&](yul::FunctionCall const& _call) -> bool
			{
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
			}},
		_expr);
}

bool yulStmtMayStoreToStorage(yul::Statement const& _stmt, yul::Dialect const& _dialect)
{
	return std::visit(
		util::GenericVisitor{
			[&](yul::ExpressionStatement const& _exprStmt) -> bool
			{ return yulExprMayStoreToStorage(_exprStmt.expression, _dialect); },
			[&](yul::Assignment const& _assignment) -> bool
			{ return _assignment.value && yulExprMayStoreToStorage(*_assignment.value, _dialect); },
			[&](yul::VariableDeclaration const& _varDecl) -> bool
			{ return _varDecl.value && yulExprMayStoreToStorage(*_varDecl.value, _dialect); },
			[&](yul::FunctionDefinition const& _funDef) -> bool
			{ return yulBlockMayStoreToStorage(_funDef.body, _dialect); },
			[&](yul::If const& _if) -> bool
			{
				return (_if.condition && yulExprMayStoreToStorage(*_if.condition, _dialect))
					   || yulBlockMayStoreToStorage(_if.body, _dialect);
			},
			[&](yul::Switch const& _switch) -> bool
			{
				if (_switch.expression && yulExprMayStoreToStorage(*_switch.expression, _dialect))
					return true;
				for (auto const& c: _switch.cases)
					if (yulBlockMayStoreToStorage(c.body, _dialect))
						return true;
				return false;
			},
			[&](yul::ForLoop const& _for) -> bool
			{
				return yulBlockMayStoreToStorage(_for.pre, _dialect)
					   || (_for.condition && yulExprMayStoreToStorage(*_for.condition, _dialect))
					   || yulBlockMayStoreToStorage(_for.post, _dialect)
					   || yulBlockMayStoreToStorage(_for.body, _dialect);
			},
			[&](yul::Break const&) -> bool { return false; },
			[&](yul::Continue const&) -> bool { return false; },
			[&](yul::Leave const&) -> bool { return false; },
			[&](yul::Block const& _block) -> bool { return yulBlockMayStoreToStorage(_block, _dialect); }},
		_stmt);
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
	bool matches = ((libName == "EnumerableSet" || libName == "EnumerableSetUpgradeable")
					&& (fnName == "add" || fnName == "remove"))
				   || ((libName == "EnumerableMap" || libName == "EnumerableMapUpgradeable")
					   && (fnName == "set" || fnName == "remove"))
				   || ((libName == "Checkpoints" || libName == "CheckpointsUpgradeable") && fnName == "push");
	// [storage-ref-alias review fix] Deliberately NOT extended to
	// Counters.(increment|reset) / DoubleEndedQueue.(pushBack|popFront|
	// clear), although the design's §4 named them: allowlisting a mutator
	// here removes the oracle's `unknown = true` guard for its callers,
	// which is only sound when the Lean VALUE MODEL actually performs the
	// receiver write. The existing entries above have real model-side
	// write-back semantics (e.g. EnumerableSet.add lowers to a full
	// mapping_set/array_push RMW via the EnumerableSetAdd synthetic-ref-
	// consumer machinery), but Counters.increment/reset and the
	// DoubleEndedQueue mutators translate to OPAQUE Unit-returning
	// builtins (`opaque internal_call_increment (p0 : Counter) : Unit` —
	// a modeled NO-OP; see Analysis.ml's builtin return-type table).
	// Summary.ml's hidden-mutator arm records the write for TOUCHED-FIELD
	// ACCOUNTING only, which would make FIDELITY-001 pass while the model
	// silently drops the mutation (verified concretely: StRSRP1._useNonce's
	// generated Lean returned `(state, current)` with no _nonces write).
	// Extending this list is only sound together with EnumerableSetAdd-
	// style model semantics for each added helper.
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
		bool isUsingForCall = receiverType && receiverType->category() != Type::Category::TypeType
							  && receiverType->category() != Type::Category::Module;
		if (isUsingForCall)
			return &memberAccess->expression();
	}
	if (!_call.arguments().empty())
		return _call.arguments().front().get();
	return nullptr;
}

// --- ERC-7201 namespaced-storage extension to the write-set oracle ---
//
// A bounded, structural recognizer in exactly the spirit of
// isKnownOzStorageRefLibraryMutator above: instead of relaxing the generic
// (and correct-by-default) "local storage pointer ⇒ unknown" rule for every
// local storage pointer, this recognizes ONE additional fixed shape — the
// OpenZeppelin ERC-7201 namespaced-storage getter — and resolves writes
// through it to the real flattened field, exactly as exportContract's own
// "Flatten namespaced storage struct fields into Storage" pass names it
// (`derivePrefix(structDef->name()) + member->name()`). Anything that does
// not match this exact shape keeps failing closed.
//
// isNamespacedStorageGetter (above, used by body export) only needs to
// confirm the STATEMENT SHAPE of a getter, because each `$.field`
// substitution it performs is scoped to one call site and one field. This
// oracle is different: once it resolves a local pointer to a field name,
// EVERY write through that pointer for the rest of the function is
// attributed to that field, so it additionally demands the getter's
// `.slot :=` target be a compile-time CONSTANT — otherwise two different
// calls to the "same" getter could alias different real storage locations,
// and folding them into one named field would be unsound.
bool isKnownNamespacedStorageGetter(FunctionDefinition const& _funcDef, StructDefinition const** _outStructDef)
{
	if (!isNamespacedStorageGetter(_funcDef, _outStructDef))
		return false;
	// Only Pattern 1 (a bare single-statement assembly block) is trusted
	// here. Pattern 2 (`bytes32 slot = fn(); assembly { $.slot := slot }`)
	// would require separately proving that intermediate local is itself a
	// compile-time constant, which this bounded recognizer does not take
	// on — it simply stays unresolved (fails closed).
	if (_funcDef.body().statements().size() != 1)
		return false;
	auto const* asmStmt = dynamic_cast<InlineAssembly const*>(_funcDef.body().statements().front().get());
	if (!asmStmt)
		return false;

	yul::Block const& root = asmStmt->operations().root();
	if (root.statements.size() != 1)
		return false;
	auto const* yulAssignment = std::get_if<yul::Assignment>(&root.statements.front());
	if (!yulAssignment || yulAssignment->variableNames.size() != 1 || !yulAssignment->value)
		return false;

	// externalReferences maps each Yul identifier used in this assembly
	// block to the Solidity declaration/suffix it resolves to — the same
	// mechanism ContractCompiler's own codegen consults to compile `.slot`/
	// `.offset` accesses.
	auto const& externalReferences = asmStmt->annotation().externalReferences;

	// LHS must be exactly this getter's own storage-located return
	// parameter's `.slot` — not some other variable's, and not `.offset`/
	// `.length`/etc.
	yul::Identifier const& lhsIdent = yulAssignment->variableNames.front();
	auto lhsIt = externalReferences.find(&lhsIdent);
	if (lhsIt == externalReferences.end() || lhsIt->second.suffix != "slot")
		return false;
	if (_funcDef.returnParameters().size() != 1
		|| lhsIt->second.declaration != _funcDef.returnParameters().front().get())
		return false;

	// RHS must be a raw Yul literal, or a reference to a Solidity
	// `constant`-qualified variable (never another external reference's
	// `.slot`/`.offset`/..., and never a computed Yul expression — solc's
	// own frontend already required a `constant` variable's initializer to
	// be compile-time-evaluable, so isConstant() is sufficient here without
	// re-deriving that ourselves).
	if (std::holds_alternative<yul::Literal>(*yulAssignment->value))
		return true;
	auto const* rhsIdent = std::get_if<yul::Identifier>(yulAssignment->value.get());
	if (!rhsIdent)
		return false;
	auto rhsIt = externalReferences.find(rhsIdent);
	if (rhsIt == externalReferences.end() || !rhsIt->second.suffix.empty())
		return false;
	auto const* rhsVarDecl = dynamic_cast<VariableDeclaration const*>(rhsIt->second.declaration);
	return rhsVarDecl && rhsVarDecl->isConstant();
}

// Called only after peelToBaseIdentifierForWriteOracle has already resolved
// a write target down to `_base`: re-walks the same Index/Member chain to
// find the MemberAccess applied DIRECTLY to `_base` — the first `.field`
// off of it. Any further Index/Member layers stacked on top (the
// `.push(...)`/`[i]` in `$.revenues.push(...)`/`$.revenues[i]`, say) are
// discarded, matching how an ordinary top-level state variable write is
// already tracked at whole-variable granularity elsewhere in this
// collector. Returns nullptr if `_base` IS the entire target — i.e. no
// MemberAccess was ever applied (e.g. reassigning the pointer itself,
// `$ = ...`) — a shape this recognizer does not attempt to resolve.
MemberAccess const* innermostMemberAccessOntoBase(Expression const& _target, Identifier const& _base)
{
	Expression const* current = &_target;
	MemberAccess const* innermost = nullptr;
	while (current != static_cast<Expression const*>(&_base))
	{
		if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(current))
		{
			current = &indexAccess->baseExpression();
			continue;
		}
		auto const* memberAccess = dynamic_cast<MemberAccess const*>(current);
		if (!memberAccess)
			return nullptr; // Defensive: unreachable given the caller's precondition.
		innermost = memberAccess;
		current = &memberAccess->expression();
	}
	return innermost;
}

// --- End ERC-7201 namespaced-storage extension ---

// --- EIP-1967 / OpenZeppelin StorageSlot getXSlot(...) write-oracle
// extension (Phase 1c of the general storage-reference-alias-tracking
// design; see the exporter's body-side note in `resolveStorageRefInitializer`
// for the full rationale) ---
//
// `StorageSlot.getAddressSlot(slot).value = x;` (and its Boolean/Bytes32/
// Uint256/Int256/String/Bytes twins) is ALREADY modeled soundly and
// generally by the OCaml frontend, independently of this exporter: Base.ml's
// `slot_helper_record_info_of_name` recognizes the exact pre-existing JSON
// shape this exporter emits for it today (`StructUpdate` whose base is an
// opaque `internal_call` to one of these getters) and lowers the write via
// `foreign_storage_write_slot(world, thisAddress, slot, value)` — a raw,
// per-(address,slot) write against the shared WorldState, for ANY slot
// expression (not just a compile-time constant). This is strictly MORE
// general than a synthetic named Storage field could be, so — unlike the
// general local storage-reference-variable alias mechanism above — this
// pattern deliberately gets NO body-side treatment from this exporter at
// all; the body export is already correct as-is.
//
// The only real gap is that this independent write-set oracle has no way to
// know that, so `recordWriteToBase` below (called with a target that peels,
// through `.value`, to a call result rather than a plain Identifier) hits
// its normal fail-closed `unknown = true` default — demoting every
// function that legitimately uses this pattern (e.g.
// TransparentUpgradeableProxy's `_setAdmin`/`_setImplementation`/
// `_changeAdmin`/`_upgradeTo`) via SOL-PLAN-FIDELITY-002, even though the
// model is already sound. This bounded structural recognizer — independent
// of, and deliberately NOT sharing state with, Base.ml's OCaml-side
// name-based `is_storage_slot_helper_name` — lets `recordWriteToBase`
// instead recognize the write as "known to be soundly modeled elsewhere,
// and not a write to any NAMED field this export declares", so it
// contributes NOTHING to `writes` (there is no field to name) and does NOT
// set `unknown`. A `delegatecall`-containing wrapper (`_upgradeToAndCall*`)
// still hits the untouched `Kind::DelegateCall ⇒ unknown` arm elsewhere in
// this collector — correct and unaffected by this extension.
bool isKnownStorageSlotHelperOracle(FunctionDefinition const& _function)
{
	if (!_function.isOrdinary())
		return false;
	if (_function.visibility() == Visibility::Public || _function.visibility() == Visibility::External)
		return false;
	if (_function.parameters().size() != 1)
		return false;
	VariableDeclaration const& param = *_function.parameters().front();
	if (param.referenceLocation() == VariableDeclaration::Location::Storage)
		return false;
	Type const* paramType = param.type();
	if (!paramType || paramType->category() != Type::Category::FixedBytes)
		return false;
	if (auto const* fixedBytesType = dynamic_cast<FixedBytesType const*>(paramType))
		if (fixedBytesType->numBytes() != 32)
			return false;

	if (_function.returnParameters().size() != 1)
		return false;
	VariableDeclaration const& retParam = *_function.returnParameters().front();
	if (retParam.referenceLocation() != VariableDeclaration::Location::Storage)
		return false;
	auto const* userDefined = dynamic_cast<UserDefinedTypeName const*>(&retParam.typeName());
	if (!userDefined)
		return false;
	auto const* structDef
		= dynamic_cast<StructDefinition const*>(userDefined->pathNode().annotation().referencedDeclaration);
	if (!structDef || structDef->members().size() != 1)
		return false;
	if (structDef->members().front()->name() != "value")
		return false;

	if (!_function.isImplemented())
		return false;
	Block const& body = _function.body();
	if (body.statements().size() != 1)
		return false;
	auto const* asmStmt = dynamic_cast<InlineAssembly const*>(body.statements().front().get());
	if (!asmStmt)
		return false;
	yul::Block const& root = asmStmt->operations().root();
	if (root.statements.size() != 1)
		return false;
	auto const* yulAssignment = std::get_if<yul::Assignment>(&root.statements.front());
	if (!yulAssignment || yulAssignment->variableNames.size() != 1 || !yulAssignment->value)
		return false;
	auto const& externalReferences = asmStmt->annotation().externalReferences;
	yul::Identifier const& lhsIdent = yulAssignment->variableNames.front();
	auto lhsIt = externalReferences.find(&lhsIdent);
	if (lhsIt == externalReferences.end() || lhsIt->second.suffix != "slot")
		return false;
	if (lhsIt->second.declaration != &retParam)
		return false;
	auto const* rhsIdent = std::get_if<yul::Identifier>(yulAssignment->value.get());
	if (!rhsIdent)
		return false;
	auto rhsIt = externalReferences.find(rhsIdent);
	if (rhsIt == externalReferences.end() || !rhsIt->second.suffix.empty())
		return false;
	return rhsIt->second.declaration == &param;
}

// [storage-ref-alias review fix] Resolve a slot-getter call argument to a
// compile-time constant u256, or nullopt. Deliberately minimal (fail
// closed): a plain number/hex literal, or an Identifier/MemberAccess chain
// of `constant` variable declarations bottoming out in such a literal —
// exactly the shape the OZ EIP-1967 slot constants use
// (`bytes32 internal constant _ADMIN_SLOT = 0xb531...;`). Anything else
// (arithmetic, keccak256 calls, non-constant variables) returns nullopt.
std::optional<u256> resolveCompileTimeSlotConstant(Expression const& _expr, size_t _depth = 0)
{
	if (_depth > 16)
		return std::nullopt;
	if (auto const* literal = dynamic_cast<Literal const*>(&_expr))
	{
		if (auto const* rational = dynamic_cast<RationalNumberType const*>(literal->annotation().type))
			if (!rational->isFractional())
				return rational->literalValue(literal);
		return std::nullopt;
	}
	Declaration const* declaration = nullptr;
	if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
		declaration = identifier->annotation().referencedDeclaration;
	else if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
		declaration = memberAccess->annotation().referencedDeclaration;
	if (auto const* varDecl = dynamic_cast<VariableDeclaration const*>(declaration))
		if (varDecl->isConstant() && varDecl->value())
			return resolveCompileTimeSlotConstant(*varDecl->value(), _depth + 1);
	return std::nullopt;
}

// [storage-ref-alias review fix] Design §5 guard 1 for the write-oracle's
// StorageSlot extension: a slot WRITE may only be treated as "soundly
// modeled elsewhere" when its slot argument is a compile-time constant
// >= 2^64. The OCaml frontend models `getXSlot(k).value = v` as a raw
// `foreign_storage_write_slot` against world[thisAddress], which is a
// SEPARATE model component from the named Storage record — if k collided
// with a declared field's slot (sequential declared slots are small
// integers < 2^64), the model would diverge from the EVM (named field
// unchanged in the model, changed on chain). Keccak-image disjointness
// against mapping/array-derived slots is the same standard-model
// assumption the named-field storage model already makes. Non-constant
// or small slots keep the pre-existing fail-closed `unknown` (slot
// writes were ALWAYS refused before this extension, so this is
// zero-regression).
bool isDisjointConstantSlotArgument(FunctionCall const& _call)
{
	if (_call.arguments().size() != 1 || !_call.arguments().front())
		return false;
	auto slotValue = resolveCompileTimeSlotConstant(*_call.arguments().front());
	if (!slotValue.has_value())
		return false;
	return *slotValue >= (u256(1) << 64);
}

// --- End EIP-1967 / StorageSlot write-oracle extension ---

struct WriteOracleCollector: ASTConstVisitor
{
	std::set<std::string> writes;
	bool unknown = false;
	std::set<FunctionDefinition const*> calleesToVisit;
	ContractDefinition const* mostDerivedContract = nullptr;
	// Local storage-pointer variables (within THIS function body) recognized
	// as ERC-7201 namespaced-storage aliases — see
	// isKnownNamespacedStorageGetter above — mapped to their field prefix
	// (`derivePrefix(structDef->name())`, the same prefix exportContract's
	// "Flatten namespaced storage struct fields into Storage" pass uses).
	// Populated by visit(VariableDeclarationStatement) below; consulted by
	// recordWriteToBase in place of failing closed on that one bounded
	// shape.
	std::map<VariableDeclaration const*, std::string> namespacedAliasPrefixes;

	// General local storage-reference-variable alias tracking, Phase 1b
	// (design §4): local storage-pointer variables (within THIS function
	// body) mapped to their MAY-write root-field candidate set. `nullopt`
	// means "bound but not resolvable"; both an untracked/absent entry
	// and a nullopt entry fail closed at `recordWriteToBase`. Entries are
	// registered ONLY at declaration sites (visit(VariableDeclaration-
	// Statement) below) — never grown at assignments: a rebound variable
	// makes the whole function `unknown` via recordWriteToBase instead
	// (see visit(Assignment) below and the envelope invariant note on
	// resolveOracleStorageRefRoots). This is intentionally its own,
	// separate, flow-INSENSITIVE root-granularity resolver — NOT the same
	// code as the flow-sensitive, full-target-shape body-side resolver
	// (`resolveStorageRefInitializer` near the top of this file); see
	// that function's/§7.5's independence requirement.
	std::map<VariableDeclaration const*, std::optional<std::set<std::string>>> aliasRootCandidates;

	std::optional<std::set<std::string>> resolveOracleStorageRefRoots(Expression const& _expr)
	{
		// [storage-ref-alias review fix] SOUNDNESS INVARIANT: this
		// resolver's acceptance envelope must be NO WIDER than the
		// body-side resolver's (resolveStorageRefInitializer near the top
		// of this file). The FIDELITY-001 cross-check between the two
		// layers is only ROOT-granular (`oracle.writes ⊆
		// transitive_touched_fields`), so any alias this oracle resolves
		// that the body export does NOT substitute leaves the body's
		// status-quo copy-lowering in place — a silently DROPPED storage
		// write — and the drop is masked from FIDELITY-001 whenever the
		// same root is also touched by some other, real modeled write
		// (e.g. `m[i].a = 1; S storage p = <body-unresolvable>; p.a = 2;`
		// was accepted with `p.a = 2` missing from the model). For that
		// reason this function deliberately does NOT peel parenthesized
		// TupleExpressions and does NOT union Conditional (ternary) arms:
		// the body-side resolver resolves neither shape, so both must
		// stay nullopt here (⇒ recordWriteToBase keeps its pre-existing
		// fail-closed `unknown = true` for writes through such binds,
		// which is exactly the pre-alias-tracking refusal). Widening this
		// resolver is only sound together with a matching body-side
		// widening.
		if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr))
			return resolveOracleStorageRefRoots(indexAccess->baseExpression());
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
			return resolveOracleStorageRefRoots(memberAccess->expression());
		if (auto const* identifier = dynamic_cast<Identifier const*>(&_expr))
		{
			auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
			if (!decl)
				return std::nullopt;
			if (decl->isStateVariable())
				return std::set<std::string>{decl->name()};
			auto it = aliasRootCandidates.find(decl);
			if (it != aliasRootCandidates.end())
				return it->second;
			return std::nullopt;
		}
		if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
		{
			if (FunctionDefinition const* target
				= resolveWriteOracleCallTarget(call->expression(), mostDerivedContract))
			{
				StructDefinition const* structDef = nullptr;
				if (isKnownNamespacedStorageGetter(*target, &structDef))
				{
					// Root-granularity oracle: attribute to the MAY-set of
					// every field the namespaced struct could touch (the
					// exporter's own body-side substitution is exact about
					// which one; this independent oracle only needs a
					// sound over-approximation).
					std::string prefix = derivePrefix(structDef->name());
					std::set<std::string> roots;
					for (auto const& member: structDef->members())
						roots.insert(prefix + member->name());
					return roots;
				}
			}
			return std::nullopt;
		}
		return std::nullopt;
	}

	void recordWriteToBase(Expression const& _target)
	{
		if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_target))
		{
			for (auto const& component: tuple->components())
				if (component)
					recordWriteToBase(*component);
			return;
		}

		// EIP-1967 / StorageSlot `getXSlot(slot).value = ...`: see
		// isKnownStorageSlotHelperOracle's doc comment above. Already
		// soundly modeled by the OCaml frontend against
		// world[thisAddress] — not a write to any NAMED field this export
		// declares, so it must contribute nothing to `writes` and must NOT
		// set `unknown`. [storage-ref-alias review fix] Only when the slot
		// argument is a compile-time constant >= 2^64
		// (isDisjointConstantSlotArgument — design §5 guard 1): the
		// world[thisAddress] slot map is a separate model component from
		// the named Storage record, so a slot that could collide with a
		// declared field's slot must keep the pre-existing fail-closed
		// `unknown` instead.
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_target))
			if (memberAccess->memberName() == "value")
				if (auto const* call = dynamic_cast<FunctionCall const*>(&memberAccess->expression()))
					if (FunctionDefinition const* target
						= resolveWriteOracleCallTarget(call->expression(), mostDerivedContract))
						if (isKnownStorageSlotHelperOracle(*target) && isDisjointConstantSlotArgument(*call))
							return;

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
			// Immutables live in the constructor's declaration-keyed
			// environment, not in the persistent Storage record tracked by
			// this oracle. The dedicated immutable_set node owns the write.
			if (varDecl->immutable())
				return;
			// EIP-1153 `transient` state variable: a tstore never touches
			// PERSISTENT storage, and this oracle answers exactly "which
			// named Storage fields may be written" (the generator's
			// FIDELITY-001 cross-check requires oracle.writes ⊆ the
			// model's touched Storage fields — a transient name here
			// would fail that check spuriously). It must not poison
			// `unknown` either (that would demote transient-only
			// writers). Transient mutation is tracked separately through
			// the dedicated transient_set statement, which the
			// generator's transient-write marker consumes.
			if (varDecl->referenceLocation() == VariableDeclaration::Location::Transient)
				return;
			writes.insert(varDecl->name());
			return;
		}
		if (varDecl->isLocalVariable() && varDecl->referenceLocation() == VariableDeclaration::Location::Storage)
		{
			// ERC-7201 extension: if this local storage pointer was
			// initialized, in this same function body, from a call
			// recognized by isKnownNamespacedStorageGetter, attribute the
			// write to the real flattened field instead of giving up.
			auto aliasIt = namespacedAliasPrefixes.find(varDecl);
			if (aliasIt != namespacedAliasPrefixes.end())
			{
				if (MemberAccess const* member = innermostMemberAccessOntoBase(_target, *base))
				{
					writes.insert(aliasIt->second + member->memberName());
					return;
				}
				// The alias pointer itself was the whole write target
				// (e.g. `$ = ...`, reassigning the pointer) rather than a
				// `.field` access through it — not a shape this
				// recognizer covers. Fall through to fail-closed below.
			}
			// General local storage-reference-variable alias tracking
			// (Phase 1b §4): a resolved candidate root set attributes the
			// write to every candidate (sound MAY-write over-
			// approximation at root granularity); an unresolvable
			// (present-but-nullopt) or altogether untracked entry keeps
			// the pre-existing fail-closed default.
			// NOTE: a bare-identifier target reaching HERE is a genuine
			// write-through, not a rebind — recordWriteToBase is invoked
			// with bare identifiers for `.push()`/`.pop()` bases and for
			// OZ-library-mutator receiver arguments (`q.push(v)`,
			// `set.add(x)` where q/set are tracked aliases). Pointer
			// REBINDS (assignment LHS) never reach this function: they
			// are intercepted, poisoned, and failed closed in
			// visit(Assignment) below.
			auto rootIt = aliasRootCandidates.find(varDecl);
			if (rootIt != aliasRootCandidates.end() && rootIt->second.has_value())
			{
				writes.insert(rootIt->second->begin(), rootIt->second->end());
				return;
			}
			// Every other local storage-pointer variable: without alias
			// analysis we cannot statically tell which state variable it
			// points at. Fail closed rather than guessing.
			unknown = true;
			return;
		}
		// Otherwise a plain memory/calldata/stack local: not a storage
		// write at all, nothing to record.
	}

	bool visit(VariableDeclarationStatement const& _stmt) override
	{
		// Namespaced storage (ERC-7201): `XStorage storage $ = _getXStorage();`
		// — mirrors the alias registration exportStmt/exportBody already do
		// for body export (namespacedStorageAliases/namespacedGetterPrefix
		// near the top of this file), but independently, using this
		// oracle's own STRICTER getter recognizer
		// (isKnownNamespacedStorageGetter) rather than trusting that pass.
		if (_stmt.declarations().size() == 1 && _stmt.declarations().front()
			&& _stmt.declarations().front()->referenceLocation() == VariableDeclaration::Location::Storage
			&& _stmt.initialValue())
		{
			VariableDeclaration const* decl = _stmt.declarations().front().get();
			bool handledAsNamespaced = false;
			if (auto const* call = dynamic_cast<FunctionCall const*>(_stmt.initialValue()))
			{
				if (FunctionDefinition const* target
					= resolveWriteOracleCallTarget(call->expression(), mostDerivedContract))
				{
					StructDefinition const* structDef = nullptr;
					if (isKnownNamespacedStorageGetter(*target, &structDef))
					{
						namespacedAliasPrefixes[decl] = derivePrefix(structDef->name());
						handledAsNamespaced = true;
					}
				}
			}
			// General local storage-reference-variable alias tracking
			// (Phase 1b §4): only when NOT already handled by the
			// (stricter, pre-existing) namespaced-getter recognizer above.
			if (!handledAsNamespaced)
				aliasRootCandidates[decl] = resolveOracleStorageRefRoots(*_stmt.initialValue());
		}
		return true;
	}

	// [storage-ref-alias review fix] Assignment-target dispatch: a BARE
	// IDENTIFIER assignment target that is a storage-located local or
	// parameter is a pointer REBIND (Solidity only allows assigning a
	// storage reference into a storage-pointer variable), NOT a storage
	// write. The body-side export NEVER tracks a rebound variable
	// (StorageRefRebindScanner's whole-function pre-pass), so writes
	// through it stay copy-lowered (dropped from the model); an earlier
	// revision of this collector "precisely" merged candidate roots on
	// rebind instead, which attributed those dropped writes to real
	// roots and let the function pass both FIDELITY gates whenever the
	// root was also touched by a genuine write (root-granularity
	// masking) — including through storage-pointer PARAMETERS, which
	// the design explicitly keeps fail-closed (Phase-2 scope). A rebind
	// therefore poisons the variable's candidate entry AND fails the
	// whole function closed (`unknown = true` dominates any earlier-in-
	// AST-order attribution through this variable, so flow-insensitivity
	// is harmless) — exactly the pre-alias-tracking refusal behavior.
	// This dispatch deliberately lives at the ASSIGNMENT visitor, not in
	// recordWriteToBase: bare identifiers are legitimate write-through
	// targets in recordWriteToBase's OTHER call contexts (push/pop
	// bases, OZ-mutator receivers).
	void handleAssignmentTarget(Expression const& _target)
	{
		if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_target))
		{
			for (auto const& component: tuple->components())
				if (component)
					handleAssignmentTarget(*component);
			return;
		}
		if (auto const* identifier = dynamic_cast<Identifier const*>(&_target))
			if (auto const* decl
				= dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration))
				if (!decl->isStateVariable() && decl->referenceLocation() == VariableDeclaration::Location::Storage)
				{
					auto it = aliasRootCandidates.find(decl);
					if (it != aliasRootCandidates.end())
						it->second = std::nullopt;
					unknown = true;
					return;
				}
		recordWriteToBase(_target);
	}

	bool visit(Assignment const& _assignment) override
	{
		handleAssignmentTarget(_assignment.leftHandSide());
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
		if (_call.annotation().kind.set() && *_call.annotation().kind != FunctionCallKind::FunctionCall)
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
WriteOracleResult computeAstWriteOracle(FunctionDefinition const& _function, ContractDefinition const& _contract)
{
	WriteOracleResult result;
	if (!_function.isImplemented())
		return result;

	std::set<FunctionDefinition const*> visitedFunctions;
	std::set<ModifierDefinition const*> visitedModifiers;
	std::vector<FunctionDefinition const*> functionWorklist{&_function};
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
					ModifierDefinition const* modDef = resolveModifierDefinition(*fn, *modifierInvocation);
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

// --- Bucket-1 exporter closure (spechunt-reject-buckets-design §1.2.1) ---
// Collect every statically-resolved internal callee whose DEFINING scope is
// outside the exporting contract's C3 linearization — library functions
// (`ContractDefinition::isLibrary()`) and file-level free functions
// (`SourceUnit` scope). The three existing emission passes in exportContract
// only cover `definedFunctions()` of the linearization, so these callees'
// call sites used to be exported with no definition anywhere in the crate
// and the generator's `validate_internal_call_catalog` failed closed on
// them. Mirrors collectSuperReferencedFunctions' worklist idiom and uses the
// SAME callee-resolution rules as computeAstWriteOracle
// (resolveWriteOracleCallTarget): only `referencedDeclaration`-backed static
// resolutions are collected — unresolved / virtual-slot / function-typed-
// value callees remain the generator's fail-closed problem, unchanged.
// Recurses into collected bodies to a fixpoint (library fns calling other
// library/free fns) and into the modifier chain of every visited function.
// Returns callees in deterministic first-discovery order (the emission pass
// preserves it, keeping artifact output stable).
std::vector<FunctionDefinition const*> collectOutOfLinearizationCallees(ContractDefinition const& _contract)
{
	struct OutOfLinearizationCalleeCollector: ASTConstVisitor
	{
		ContractDefinition const& mostDerived;
		std::vector<FunctionDefinition const*> found;

		explicit OutOfLinearizationCalleeCollector(ContractDefinition const& _mostDerived): mostDerived(_mostDerived) {}

		bool visit(FunctionCall const& _call) override
		{
			noteTarget(resolveWriteOracleCallTarget(_call.expression(), &mostDerived));
			return true;
		}

		bool visit(Identifier const& _identifier) override
		{
			noteReferencedFunction(_identifier.annotation().referencedDeclaration);
			return true;
		}

		bool visit(MemberAccess const& _member) override
		{
			noteReferencedFunction(_member.annotation().referencedDeclaration);
			return true;
		}

		void noteReferencedFunction(Declaration const* _declaration)
		{
			if (auto const* function = dynamic_cast<FunctionDefinition const*>(_declaration))
				noteTarget(function);
		}

		// User-defined operators (UDVT `using {f as +} for T global`) are
		// dispatched by exportExpr through the type checker's statically
		// bound `userDefinedFunction` annotation, not through a FunctionCall
		// node — collect those targets too, or the emitted call sites would
		// dangle. Operator definitions are file-level free functions by
		// language rule, so the SourceUnit arm of noteTarget covers them.
		bool visit(BinaryOperation const& _operation) override
		{
			if (_operation.annotation().userDefinedFunction.set())
				noteTarget(*_operation.annotation().userDefinedFunction);
			return true;
		}

		bool visit(UnaryOperation const& _operation) override
		{
			if (_operation.annotation().userDefinedFunction.set())
				noteTarget(*_operation.annotation().userDefinedFunction);
			return true;
		}

		void noteTarget(FunctionDefinition const* target)
		{
			if (!target || !target->isOrdinary() || !target->isImplemented())
				return;
			bool outOfLinearization = false;
			if (auto const* scopeContract = dynamic_cast<ContractDefinition const*>(target->scope()))
			{
				auto const& hierarchy = mostDerived.annotation().linearizedBaseContracts;
				// INTERNAL-visibility library functions only: public/external
				// library functions have delegatecall semantics and are already
				// handled by the generator's sibling-artifact import machinery
				// (Frontend.ml augment_crate_with_sibling_internal_functions),
				// which carries the extension-method receiver provenance the
				// exporter's using-for lowering only produces for internal
				// callees (visibility() <= Internal gates at the call sites).
				// Embedding a public library fn here shadows that import and
				// regressed lib-storage-public-2 (receiver provenance lost).
				outOfLinearization = scopeContract->isLibrary() && target->visibility() <= Visibility::Internal
									 && std::find(hierarchy.begin(), hierarchy.end(), scopeContract) == hierarchy.end();
			}
			else if (dynamic_cast<SourceUnit const*>(target->scope()))
				outOfLinearization = true;
			if (outOfLinearization)
				found.push_back(target);
		}
	};

	std::vector<FunctionDefinition const*> collected;
	std::set<FunctionDefinition const*> collectedSet;
	std::set<FunctionDefinition const*> visitedFunctions;
	std::set<ModifierDefinition const*> visitedModifiers;
	std::vector<FunctionDefinition const*> functionWorklist;
	std::vector<ModifierDefinition const*> modifierWorklist;

	// Seed with every implemented function body the existing passes will
	// export: definedFunctions() of the whole linearization includes ordinary
	// functions, constructors, receive and fallback (the passes filter by
	// isOrdinary/visibility themselves; a callee referenced only from a body
	// that later fails export is merely dead-but-defined in the artifact).
	for (ContractDefinition const* base: _contract.annotation().linearizedBaseContracts)
		for (FunctionDefinition const* function: base->definedFunctions())
			if (function && function->isImplemented())
				functionWorklist.push_back(function);

	while (!functionWorklist.empty() || !modifierWorklist.empty())
	{
		if (!functionWorklist.empty())
		{
			FunctionDefinition const* fn = functionWorklist.back();
			functionWorklist.pop_back();
			if (!fn || visitedFunctions.count(fn) || !fn->isImplemented())
				continue;
			visitedFunctions.insert(fn);

			OutOfLinearizationCalleeCollector collector{_contract};
			fn->body().accept(collector);
			for (FunctionDefinition const* target: collector.found)
				if (collectedSet.insert(target).second)
				{
					collected.push_back(target);
					functionWorklist.push_back(target);
				}

			for (auto const& modifierInvocation: fn->modifiers())
			{
				ModifierDefinition const* modDef = resolveModifierDefinition(*fn, *modifierInvocation);
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

		OutOfLinearizationCalleeCollector collector{_contract};
		mod->body().accept(collector);
		for (FunctionDefinition const* target: collector.found)
			if (collectedSet.insert(target).second)
			{
				collected.push_back(target);
				functionWorklist.push_back(target);
			}
	}

	return collected;
}

/// Solidity constants are normally substituted at expression-use sites and
/// therefore do not otherwise have a runtime declaration in SolCore. Inline
/// assembly is different: solc's externalReferences interface names the exact
/// VariableDeclaration ID, and the frontend validates that ID against the
/// declarations visible at the assembly statement. Preserve each captured
/// constant as an initializer-backed lexical binding instead of asking the
/// consumer to reconstruct it from a name or fabricate a value.
std::vector<VariableDeclaration const*> inlineAssemblyConstantDeclarations(FunctionDefinition const& _function)
{
	struct Collector: ASTConstVisitor
	{
		std::map<std::string, VariableDeclaration const*> declarations;

		bool visit(InlineAssembly const& _assembly) override
		{
			for (auto const& entry: _assembly.annotation().externalReferences)
			{
				auto const* declaration
					= dynamic_cast<VariableDeclaration const*>(entry.second.declaration);
				if (!declaration || !declaration->isConstant())
					continue;
				if (!declaration->value())
					throw UnsupportedSolCore(
						"Inline assembly references constant '" + declaration->name()
						+ "' without a compiler-resolved initializer.");

				std::string declarationId = std::to_string(declaration->id());
				auto [it, inserted] = declarations.emplace(declarationId, declaration);
				if (!inserted && it->second != declaration)
					throw UnsupportedSolCore(
						"Inline assembly constant declaration ID is not unique.");
			}
			return true;
		}
	};

	Collector collector;
	_function.body().accept(collector);
	for (auto const& modifierInvocation: _function.modifiers())
	{
		ModifierDefinition const* modifierDefinition
			= resolveModifierDefinition(_function, *modifierInvocation);
		if (modifierDefinition && modifierDefinition->isImplemented())
			modifierDefinition->body().accept(collector);
	}

	std::vector<VariableDeclaration const*> result;
	result.reserve(collector.declarations.size());
	for (auto const& [declarationId, declaration]: collector.declarations)
	{
		(void) declarationId;
		result.push_back(declaration);
	}
	return result;
}

Json exportBody(FunctionDefinition const& _function)
{
	// Save and clear per-function namespaced storage aliases
	// (the getter prefix map persists across functions)
	auto savedAliases = namespacedStorageAliases;
	namespacedStorageAliases.clear();

	// General storage-reference-variable alias tracking (Phase 1a): fresh
	// per-function state, plus the whole-function rebind pre-pass (see the
	// deviation note on storageRefAliasTargets above).
	//
	// [storage-ref-alias review fix] The pre-pass must cover every AST
	// body that will be exported under THIS alias scope — that is the
	// function body AND every resolved modifier body (expandModifiers
	// below exports modifier bodies with the same storageRefAliasTargets
	// state active). Scanning only the function body left a modifier that
	// rebinds a storage-ref local un-scanned: the bind would be tracked,
	// the rebind exported as a plain local assign, and every subsequent
	// use substituted with the STALE pre-rebind target (writes to the
	// wrong storage slot).
	StorageRefAliasScope storageRefAliasScope;
	{
		StorageRefRebindScanner rebindScanner;
		// [P0 fail-closed] The array-shrink pre-pass has exactly the same
		// scope requirement as the rebind pre-pass: every AST body exported
		// under this alias scope, i.e. the function body AND every resolved
		// modifier body (a `pop()` inside a modifier shrinks the array just
		// as effectively as one in the body).
		StorageRefShrinkScanner shrinkScanner;
		// [P0 fail-closed: E0(a')] Same scope requirement again: a write
		// through a storage-pointer local can just as well live in a
		// modifier body exported under this alias scope.
		StorageRefWriteThroughScanner writeThroughScanner;
		_function.body().accept(rebindScanner);
		_function.body().accept(shrinkScanner);
		_function.body().accept(writeThroughScanner);
		for (auto const& modifierInvocation: _function.modifiers())
		{
			ModifierDefinition const* modifierDefinition = resolveModifierDefinition(_function, *modifierInvocation);
			if (modifierDefinition && modifierDefinition->isImplemented())
			{
				modifierDefinition->body().accept(rebindScanner);
				modifierDefinition->body().accept(shrinkScanner);
				modifierDefinition->body().accept(writeThroughScanner);
			}
		}
		storageRefNeverTrack = rebindScanner.neverTrack;
		storageRefShrinkInfo = shrinkScanner.info;
		storageRefWriteThroughLocals = writeThroughScanner.writtenThrough;
		// [E2(b)] Single-assignment (declare-then-assign) storage-pointer
		// locals: the assignment is the binding initializer. Their bind-site
		// resolution and the deferred E0(a') refusal both live in
		// exportAssignment.
		storageRefSingleAssignBindable = rebindScanner.singleAssignBindable();
	}
	activeResidualStorageRefReturn = isResidualStorageRefFunction(_function);
	for (auto const& parameter: _function.parameters())
		if (isStructuralStorageRefParameter(_function, parameter.get()))
			storageRefValueLocals.insert(parameter.get());


	Json body;
	try
	{
		// Refuse unsupported implicit bytesN widening and capture each
		// byte/string literal's declared target before expression export.
		// Reset per function so a pointer from an earlier AST cannot leak.
		{
			canonicalLiteralTargets.clear();
			NarrowBytesWideningScanner narrowBytesScanner;
			_function.body().accept(narrowBytesScanner);
			for (auto const& modifierInvocation: _function.modifiers())
			{
				ModifierDefinition const* modifierDefinition
					= resolveModifierDefinition(_function, *modifierInvocation);
				if (modifierDefinition && modifierDefinition->isImplemented())
				{
					modifierDefinition->body().accept(narrowBytesScanner);
					// Modifier-invocation ARGUMENTS are outside the function
					// body AST: scan each against the modifier's declared
					// parameter type (same implicit-conversion site class).
					if (modifierInvocation->arguments())
					{
						auto const& params = modifierDefinition->parameters();
						auto const& margs = *modifierInvocation->arguments();
						if (params.size() == margs.size())
							for (size_t i = 0; i < margs.size(); ++i)
								if (margs[i])
								{
									margs[i]->accept(narrowBytesScanner);
									NarrowBytesWideningScanner::checkFlow(*margs[i], params[i]->type());
								}
					}
				}
			}
		}
		body = exportStmt(_function.body());
		body = expandModifiers(_function, std::move(body));
		auto assemblyConstants = inlineAssemblyConstantDeclarations(_function);
		if (!assemblyConstants.empty())
		{
			Json statements = Json::array();
			for (VariableDeclaration const* declaration: assemblyConstants)
			{
				Json binding = Json::object();
				binding["kind"] = "let";
				binding["sourceDeclarationId"] = std::to_string(declaration->id());
				binding["name"] = declaration->name();
				binding["type"] = exportTypeName(declaration->typeName());
				binding["value"] = exportExpr(*declaration->value());
				statements.emplace_back(std::move(binding));
			}
			for (auto const& statement: body["statements"])
				statements.emplace_back(statement);
			body["statements"] = std::move(statements);
		}
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

	if (!activeRawSlotStorageRefParams.empty())
	{
		Json prelude = Json::array();
		for (auto const& parameter: _function.parameters())
			if (activeRawSlotStorageRefParams.count(parameter.get()))
			{
				Json rawSlot = Json::object();
				rawSlot["kind"] = "storage_ref_raw_slot";
				rawSlot["referentType"] = exportResolvedType(parameter->annotation().type, true);
				rawSlot["slot"] = localExpr(parameter->name());

				Json letReference = Json::object();
				letReference["kind"] = "let";
				letReference["sourceDeclarationId"] = Json();
				letReference["name"] = parameter->name();
				letReference["type"] = storageRefWireType(parameter->annotation().type);
				letReference["value"] = std::move(rawSlot);
				prelude.emplace_back(std::move(letReference));
			}
		for (auto const& statement: body["statements"])
			prelude.emplace_back(statement);
		body["statements"] = std::move(prelude);
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
				letStmt["sourceDeclarationId"] = std::to_string(retParam->id());
				letStmt["name"] = retParam->name();
				if (Json location = dataLocationEntry(*retParam); !location.is_null())
					letStmt["location"] = std::move(location);
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
			else if (_function.returnParameters().size() == 1 && !_function.returnParameters().front()->name().empty())
			{
				// Implicit return of the named return variable
				Json local = Json::object();
				local["kind"] = "local";
				local["name"] = _function.returnParameters().front()->name();
				ret["value"] = local;
			}
			else if (_function.returnParameters().size() == 1)
			{
				// Unnamed return parameter — preserve the resolved type and its
				// memory/storage location in the compiler-inserted default.
				ret["value"] = typedDefaultValueForResolvedType(
					_function.returnParameters().front()->annotation().type);
			}
			else
			{
				Json tuple = Json::object();
				tuple["kind"] = "tuple";
				tuple["elements"] = Json::array();
				for (auto const& retParam: _function.returnParameters())
				{
					if (!retParam->name().empty())
					{
						Json local = Json::object();
						local["kind"] = "local";
						local["name"] = retParam->name();
						tuple["elements"].emplace_back(std::move(local));
					}
					else
						tuple["elements"].emplace_back(
							typedDefaultValueForResolvedType(retParam->annotation().type));
				}
				ret["value"] = std::move(tuple);
			}
			body["statements"].emplace_back(ret);
		}
	}
	namespacedStorageAliases = savedAliases;
	return body;
}

/// `_spec`: when non-null, export a SPECIALIZED sibling of `_function`
/// instead of its plain form (bounded defunctionalization — see the block
/// preceding exportExpr's definition): the bound parameters are erased from
/// the signature, the synthetic name bypasses `exportedFunctionName`, and
/// `activeFnPtrBindings` is installed for the duration of the body export so
/// every call through a bound parameter resolves to its static target.
Json exportFunction(
	FunctionDefinition const& _function,
	ContractDefinition const& _contract,
	bool _isInternal = false,
	FnPtrSpecializationRequest const* _spec = nullptr,
	bool _storageRefCompanion = false)
{
	if (!_function.isOrdinary() || !_function.isImplemented())
		throw UnsupportedSolCore("Only ordinary implemented functions are supported.");
	if (!_isInternal
		&& !(_function.visibility() == Visibility::Public || _function.visibility() == Visibility::External))
		throw UnsupportedSolCore("Only public/external functions are supported.");
	// Note: multiple return values are exported as a tuple return type.

	std::set<VariableDeclaration const*> boundParams;
	if (_spec)
		for (FnPtrBinding const& binding: _spec->bindings)
			boundParams.insert(binding.param);
	auto const* definingContract = dynamic_cast<ContractDefinition const*>(_function.scope());
	bool const rawSlotAbiEntry
		= !_isInternal && definingContract == &_contract && isPublicLibraryStructuralStorageFunction(_function);
	bool const abiVisible = !_isInternal && !_storageRefCompanion;
	bool const abiForLibrary = abiVisible && definingContract && definingContract->isLibrary();
	std::set<VariableDeclaration const*> rawSlotParams;
	if (rawSlotAbiEntry)
		for (auto const& parameter: _function.parameters())
			if (isStorageRefParameter(parameter.get()))
				rawSlotParams.insert(parameter.get());


	Json result = Json::object();
	if (_storageRefCompanion)
	{
		result["name"] = storageRefInternalEntryName(_function);
		result["originalName"] = _function.name().empty() ? "_unnamed" : _function.name();
	}
	else if (_spec)
	{
		// Bypasses exportedFunctionName: the same FunctionDefinition can have
		// MULTIPLE specialized siblings (one per distinct binding tuple), so
		// there is no single 1:1 name for it to memoize.
		result["name"] = _spec->specializedName;
		result["originalName"] = _function.name().empty() ? "_unnamed" : _function.name();
	}
	else
	{
		result["name"] = exportedFunctionName(_function);
		if (result["name"] != (_function.name().empty() ? "_unnamed" : _function.name()))
			result["originalName"] = _function.name().empty() ? "_unnamed" : _function.name();
	}
	if (_isInternal || _storageRefCompanion)
		result["visibility"] = "internal";
	if (!_function.modifiers().empty())
		result["has_modifiers"] = true;
	result["params"] = Json::array();
	for (auto const& parameter: _function.parameters())
	{
		if (_spec && boundParams.count(parameter.get()))
			// Erased from the specialized signature — bound to a static
			// target instead, resolved inside the body via
			// activeFnPtrBindings.
			continue;
		if (isStructuralStorageRefParameter(_function, parameter.get()))
		{
			Json param = Json::object();
			param["sourceDeclarationId"] = std::to_string(parameter->id());
			param["name"] = parameter->name().empty() ? ("arg" + std::to_string(parameter->id())) : parameter->name();
			param["type"] = rawSlotParams.count(parameter.get()) ? Json("u256")
																 : storageRefWireType(parameter->annotation().type);
			if (Json location = dataLocationEntry(*parameter); !location.is_null())
				param["location"] = std::move(location);
			if (abiVisible)
				param["abi"] = exportAbiDescriptor(
					parameter->name(), parameter->annotation().type, abiForLibrary);
			result["params"].emplace_back(std::move(param));
			continue;
		}
		if (abiVisible)
			result["params"].emplace_back(exportParam(*parameter, true, abiForLibrary));
		else
		{
			try
			{
				result["params"].emplace_back(exportParam(*parameter));
			}
			catch (...)
			{
				// Preserve the pre-cutover execution carrier for internal-only
				// declarations whose type cannot be rendered by SolCore. No ABI
				// descriptor is invented for this fallback.
				Json param = Json::object();
				param["sourceDeclarationId"] = std::to_string(parameter->id());
				param["name"] = parameter->name().empty() ? ("arg" + std::to_string(parameter->id())) : parameter->name();
				param["type"] = Json("u256");
				result["params"].emplace_back(param);
			}
		}
	}
	if (_function.returnParameters().empty())
		result["return"] = Json("unit");
	else if (_function.returnParameters().size() == 1)
	{
		if (isResidualStorageRefFunction(_function))
			result["return"] = storageRefWireType(_function.returnParameters().front()->annotation().type);
		else
		{
			if (abiVisible)
				result["return"] = exportTypeName(_function.returnParameters().front()->typeName());
			else
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
			if (abiVisible)
				tupleType["elements"].emplace_back(exportTypeName(retParam->typeName()));
			else
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
		}
		result["return"] = tupleType;
	}
	if (abiVisible)
	{
		result["returnAbi"] = Json::array();
		for (auto const& retParam: _function.returnParameters())
			result["returnAbi"].emplace_back(exportAbiDescriptor(
				retParam->name(), retParam->annotation().type, abiForLibrary));
	}
	// Additive schema field, symmetric with exportParam's `location`: the
	// RESOLVED data locations of the RETURN parameters, aligned
	// index-for-index with the declared return parameter list (a one-element
	// array for a single return value; one entry per element of the exported
	// tuple type for several; value-typed positions are null). Without it a
	// consumer cannot distinguish `returns (S storage)` from
	// `returns (S memory)` — both export the bare erased type — and the two
	// have DIFFERENT external ABIs for a public library function: a storage
	// reference crosses the delegatecall boundary as a uint256 SLOT WORD,
	// while a memory value is ABI-encoded (design E4a). Body-level inference
	// cannot recover this (the legal `returns (S memory) { return s; }` copy
	// shape has a storage-rooted return root too), so the exporter must emit
	// it. Taken from the type checker's annotated type, never from the
	// declaration keyword alone. Omitted entirely when NO return position is
	// reference/mapping-typed, so consumers that predate the field are
	// unaffected; consumers must treat an ABSENT location on a
	// reference-typed return as unknown and refuse, never assume memory.
	{
		Json returnLocations = Json::array();
		bool anyLocation = false;
		for (auto const& retParam: _function.returnParameters())
		{
			Json entry = retParam ? dataLocationEntry(*retParam) : Json();
			if (!entry.is_null())
				anyLocation = true;
			returnLocations.emplace_back(std::move(entry));
		}
		if (anyLocation)
			result["returnLocations"] = std::move(returnLocations);
	}
	{
		// Install activeFnPtrBindings for exactly the duration of this
		// body's export (RAII; exception-safe) so every call through a
		// bound function-typed parameter resolves to its static target
		// instead of hitting the indirect-call fail-closed path.
		std::optional<FnPtrBindingScope> bindingScope;
		if (_spec)
			bindingScope.emplace(*_spec);
		auto savedRawSlotParams = activeRawSlotStorageRefParams;
		activeRawSlotStorageRefParams = rawSlotParams;
		try
		{
			result["body"] = exportBody(_function);
		}
		catch (...)
		{
			activeRawSlotStorageRefParams = std::move(savedRawSlotParams);
			throw;
		}
		activeRawSlotStorageRefParams = std::move(savedRawSlotParams);
	}
	// Layer 2 (SOL-PLAN-FIDELITY): independent AST write-set cross-check.
	// Computed even when the body export above succeeded — it is a second
	// opinion, not just a failure fallback. NOTE: not (yet) binding-aware —
	// an indirect call through a bound function-typed parameter still
	// degrades to `unknown=true` here exactly as before specialization
	// existed (see SOLCORE_FNPTR_DEFUNCTIONALIZATION_DESIGN §3.7: a
	// binding-aware oracle is a precision improvement with NO observable
	// effect on the current corpus, since every affected callee also has an
	// independent local-storage-pointer parameter that already forces
	// `unknown=true`; deferred rather than risking a mistake in this
	// soundness-critical component for zero corpus-visible benefit today).
	result["ast_write_oracle"] = astWriteOracleJson(_function, _contract);
	if (_spec)
	{
		// Additive, informational provenance field — purely descriptive,
		// never consumed for correctness. Bindings are already fully baked
		// into the specialized body/signature above.
		Json specInfo = Json::object();
		specInfo["of"] = _spec->baseExportedName;
		Json bindingsJson = Json::array();
		for (FnPtrBinding const& binding: _spec->bindings)
		{
			Json b = Json::object();
			b["param"]
				= binding.param->name().empty() ? ("arg" + std::to_string(binding.param->id())) : binding.param->name();
			b["target"] = binding.targetExportedName;
			bindingsJson.push_back(std::move(b));
		}
		specInfo["bindings"] = std::move(bindingsJson);
		result["fnptr_specialization"] = std::move(specInfo);
	}
	return result;
}

std::vector<ASTPointer<Expression>> const* constructorArguments(ASTNode const& _node)
{
	if (auto const* inheritance = dynamic_cast<InheritanceSpecifier const*>(&_node))
		return inheritance->arguments();
	if (auto const* invocation = dynamic_cast<ModifierInvocation const*>(&_node))
		return invocation->arguments();
	throw UnsupportedSolCore("Base-constructor argument annotation has an unsupported AST owner.");
}

void appendConstructorBodyStatements(
	Json& _statements, Json const& _body, std::string const& _contractName, bool _includeTerminalReturn)
{
	if (!_body.is_object() || _body.value("kind", ""s) != "block" || !_body.contains("statements")
		|| !_body["statements"].is_array())
		throw UnsupportedSolCore(
			"Constructor body for '" + _contractName + "' could not be exported faithfully: "
			+ (_body.is_object() && _body.contains("error") && _body["error"].is_string()
				   ? _body["error"].get<std::string>()
				   : "body did not have the required block shape"));
	Json const& bodyStatements = _body["statements"];
	for (size_t i = 0; i < bodyStatements.size(); ++i)
		if (
			_includeTerminalReturn || i + 1 != bodyStatements.size() || !bodyStatements[i].is_object()
			|| bodyStatements[i].value("kind", ""s) != "return"
		)
			_statements.emplace_back(bodyStatements[i]);
}
void appendStateVariableInitializers(Json& _statements, ContractDefinition const& _contract)
{
	for (VariableDeclaration const* variable: _contract.stateVariables())
	{
		if (variable->isConstant() || !variable->value())
			continue;
		Json statement = Json::object();
		if (variable->immutable())
			statement = immutableSet(*variable, exportExpr(*variable->value()));
		else if (isTransientStateVar(variable))
		{
			statement["kind"] = "transient_set";
			statement["field"] = variable->name();
			statement["value"] = exportExpr(*variable->value());
		}
		else
		{
			statement["kind"] = "storage_set";
			statement["field"] = variable->name();
			statement["value"] = exportExpr(*variable->value());
		}
		_statements.emplace_back(std::move(statement));
	}
}

Json exportConstructorChainBody(ContractDefinition const& _mostDerived)
{
	try
	{
		auto const& derivedFirst = _mostDerived.annotation().linearizedBaseContracts;
		std::map<VariableDeclaration const*, std::string> argumentTemps;

		auto buildFrame = [&](auto&& self, size_t _index) -> Json
		{
			ContractDefinition const* current = derivedFirst.at(_index);
			Json frame = Json::object();
			frame["kind"] = "block";
			frame["statements"] = Json::array();
			Json& statements = frame["statements"];

			if (_index != 0)
				if (FunctionDefinition const* constructor = current->constructor())
					for (auto const& parameter: constructor->parameters())
					{
						auto temp = argumentTemps.find(parameter.get());
						if (temp == argumentTemps.end())
							throw UnsupportedSolCore(
								"No evaluated argument was available for base constructor parameter '"
								+ parameter->name() + "' of '" + current->name() + "'.");
						Json binding = Json::object();
						binding["kind"] = "let";
						binding["sourceDeclarationId"] = std::to_string(parameter->id());
						binding["name"] = parameter->name().empty()
											  ? ("constructor_arg" + std::to_string(parameter->id()))
											  : parameter->name();
						binding["type"] = exportTypeName(parameter->typeName());
						binding["value"] = localExpr(temp->second);
						statements.emplace_back(std::move(binding));
					}

			for (ContractDefinition const* target: derivedFirst)
			{
				FunctionDefinition const* targetConstructor = target->constructor();
				if (!targetConstructor)
					continue;
				auto annotated
					= _mostDerived.annotation().baseConstructorArguments.find(targetConstructor);
				if (annotated == _mostDerived.annotation().baseConstructorArguments.end()
					|| !current->location().contains(annotated->second->location()))
					continue;
				auto const* arguments = constructorArguments(*annotated->second);
				if (!arguments || arguments->size() != targetConstructor->parameters().size())
					throw UnsupportedSolCore(
						"Base constructor argument count did not match the compiler-resolved parameter list for '"
						+ target->name() + "'.");
				for (size_t i = 0; i < arguments->size(); ++i)
				{
					VariableDeclaration const* parameter = targetConstructor->parameters()[i].get();
					std::string tempName = "__solcore_ctor_arg_" + std::to_string(parameter->id());
					Json evaluated = Json::object();
					evaluated["kind"] = "let";
					evaluated["sourceDeclarationId"] = Json();
					evaluated["name"] = tempName;
					evaluated["type"] = exportTypeName(parameter->typeName());
					evaluated["value"] = exportExpr(*arguments->at(i));
					statements.emplace_back(std::move(evaluated));
					argumentTemps[parameter] = std::move(tempName);
				}
			}

			if (_index + 1 < derivedFirst.size())
				statements.emplace_back(self(self, _index + 1));

			appendStateVariableInitializers(statements, *current);
			if (FunctionDefinition const* constructor = current->constructor())
			{
				if (!constructor->isImplemented())
					throw UnsupportedSolCore(
						"Base constructor for '" + current->name() + "' has no exportable body.");
				appendConstructorBodyStatements(
					statements, exportBody(*constructor), current->name(), _index == 0);
			}
			else if (_index == 0)
				statements.emplace_back(Json{{"kind", "return"}, {"value", Json{{"kind", "unit"}}}});
			return frame;
		};

		if (derivedFirst.empty() || derivedFirst.front() != &_mostDerived)
			throw UnsupportedSolCore("Constructor C3 linearization did not start at the most-derived contract.");
		return buildFrame(buildFrame, 0);
	}
	catch (...)
	{
		std::string reason = "unknown constructor-chain export failure";
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
		}
		return Json{{"kind", "unsupported_body"}, {"error", reason}};
	}
}

Json constructorChainWriteOracle(ContractDefinition const& _contract)
{
	WriteOracleResult combined;
	for (ContractDefinition const* current: _contract.annotation().linearizedBaseContracts)
	{
		for (VariableDeclaration const* variable: current->stateVariables())
			if (!variable->isConstant() && !variable->immutable() && variable->value())
				combined.writes.insert(variable->name());
		if (FunctionDefinition const* constructor = current->constructor())
		{
			WriteOracleResult oracle = computeAstWriteOracle(*constructor, _contract);
			combined.writes.insert(oracle.writes.begin(), oracle.writes.end());
			combined.unknown = combined.unknown || oracle.unknown;
		}
	}
	Json result = Json::object();
	result["writes"] = Json::array();
	for (std::string const& name: combined.writes)
		result["writes"].emplace_back(name);
	result["unknown"] = combined.unknown;
	return result;
}

Json exportImplicitConstructor(ContractDefinition const& _contract)
{
	Json result = Json::object();
	result["name"] = "constructor";
	result["declarationId"] = std::to_string(_contract.id());
	result["sourceLocation"] = sourceLocation(_contract.location());
	result["paramsAbi"] = Json::array();
	result["params"] = Json::array();
	result["return"] = Json("unit");
	result["returnAbi"] = Json::array();
	result["body"] = exportConstructorChainBody(_contract);
	result["ast_write_oracle"] = constructorChainWriteOracle(_contract);
	return result;
}

Json exportConstructor(FunctionDefinition const& _function, ContractDefinition const& _contract)
{
	if (!_function.isConstructor() || !_function.isImplemented())
		throw UnsupportedSolCore("Only implemented constructors are supported.");

	Json result = Json::object();
	result["name"] = "constructor";
	result["declarationId"] = std::to_string(_function.id());
	result["sourceLocation"] = sourceLocation(_function.location());
	result["paramsAbi"] = exportConstructorParamsAbi(_function);
	result["params"] = Json::array();
	for (auto const& parameter: _function.parameters())
		result["params"].emplace_back(exportParam(*parameter, true, false));
	result["return"] = Json("unit");
	result["returnAbi"] = Json::array();
	result["body"] = exportConstructorChainBody(_contract);
	result["ast_write_oracle"] = constructorChainWriteOracle(_contract);
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
		entry["kind"] = stateVar->referenceLocation() == VariableDeclaration::Location::Transient
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
	// Inherited closure, mirroring the `events` export: every event the
	// contract's export declares (incl. base-contract and used library
	// events) gets an origin entry, so a consumer joining events to origins
	// by name never hits a missing key. Source locations point at the
	// DECLARING contract — correct provenance for an inherited declaration.
	for (EventDefinition const* event: _contract.interfaceEvents(false))
	{
		Json entry = Json::object();
		// Disambiguated name for BOTH fields: `originId` is a key, and two
		// overloaded `E`s would otherwise mint two entries with the identical
		// `event:E` id. Non-overloaded events keep their bare name, so no
		// existing origin id changes.
		entry["originId"] = "event:" + exportedEventName(*event);
		entry["kind"] = "event";
		entry["name"] = exportedEventName(*event);
		entry["astId"] = event->id();
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
		std::string sourcePath = importDirective->annotation().absolutePath.set()
									 ? *importDirective->annotation().absolutePath
									 : std::string{};
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
	contractScopedSuperAliases.clear();
	stableSyntheticNodeIds.clear();
	nextStableSyntheticNodeId = 0;
	unboundVirtualSlotNames.clear();
	// Internal-function-used-as-a-value (bounded defunctionalization) state
	// — see the block preceding exportExpr's definition.
	fnPtrSpecializationsByName.clear();
	fnPtrSpecializationQueue.clear();
	activeFnPtrBindings.clear();
	fnPtrCalleeAdmissibleMemo.clear();
	internalFnTablesByFingerprint.clear();
	internalFnContractContainsAssembly = false;
	internalFnAssemblyTouchesValue = false;

	std::vector<NamespacedStorageGetter> namespacedGetters;
	// Scan all functions in the contract hierarchy for namespaced storage getters
	for (FunctionDefinition const* function: contract.definedFunctions())
	{
		StructDefinition const* structDef = nullptr;
		if (isNamespacedStorageGetter(*function, &structDef))
		{
			std::string prefix = derivePrefix(structDef->name());
			std::string fieldName = deriveSubStorageFieldName(structDef->name());
			namespacedGetters.push_back({function, structDef, prefix, fieldName});
			namespacedGetterPrefixes[function] = prefix;
			// Track which prefixed fields are mappings
			for (auto const& member: structDef->members())
			{
				if (dynamic_cast<Mapping const*>(&member->typeName()))
					namespacedMappingFields.insert(prefix + member->name());
			}
		}
	}
	// Also scan base contracts
	for (auto const* baseContract: contract.annotation().linearizedBaseContracts)
	{
		if (baseContract == &contract)
			continue;
		for (FunctionDefinition const* function: baseContract->definedFunctions())
		{
			StructDefinition const* structDef = nullptr;
			if (isNamespacedStorageGetter(*function, &structDef))
			{
				// Avoid duplicates
				bool alreadyFound = false;
				for (auto const& g: namespacedGetters)
					if (g.function->name() == function->name())
						alreadyFound = true;
				if (!alreadyFound)
				{
					std::string prefix = derivePrefix(structDef->name());
					std::string fieldName = deriveSubStorageFieldName(structDef->name());
					namespacedGetters.push_back({function, structDef, prefix, fieldName});
					namespacedGetterPrefixes[function] = prefix;
					for (auto const& member: structDef->members())
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
	//
	// Deliberately NOT bumped for internal-function-used-as-a-value support
	// (bounded defunctionalization; see the block preceding exportExpr's
	// definition): no OCaml-side consumer reads this field (verified by
	// grep), and bumping a single global version string here would touch
	// EVERY artifact's bytes on next regeneration -- destroying the far more
	// valuable "byte-identical for every contract with no function-typed
	// parameters" signal a corpus-wide diff otherwise gives for free. The
	// feature is already self-describing in the artifact: new
	// `<name>__fnptr__<param>__<target>` internal_functions entries, fewer
	// "used as a value" unsupported_body markers, and (only on functions
	// that use it) the additive "fnptr_specialization" provenance field.
	solcore["solcoreVersion"] = "0.2.0";
	solcore["solidityVersion"] = VersionString;
	solcore["featureFlags"] = featureFlags();
	Json metadata = exporterMetadata(_contractName, _compilerStack.viaIR());
	for (auto const& [key, value]: metadata.items())
		solcore[key] = value;
	solcore["crate_name"] = contract.name();
	solcore["deployedCodeSize"] = solcoreDeployedCodeSize(contract);
	solcore["deployedCodeSizeMode"] = "semantic-nonzero";
	solcore["source_imports"] = exportSourceImports(contract.sourceUnit());

	Json typeDecls = Json::array();
	Json subStorageGetters = Json::array();
	Json storageFields = Json::array();
	// EIP-1153 `transient`-location state variables: exported as a SEPARATE
	// crate-level list (name/type/slot), never as Storage fields — the value
	// lives in the transaction-scoped transient slot space
	// (ExecState.transient runtime-side), with different reset semantics
	// and a different address space than persistent storage.
	Json transientFields = Json::array();
	std::set<std::string> exportedFieldNames;
	std::vector<VariableDeclaration const*> immutableDeclarations;
	std::set<std::string> exportedImmutableIds;

	// Helper lambda to export state variables from a contract definition
	auto exportStateVars = [&](ContractDefinition const* source)
	{
		for (VariableDeclaration const* stateVar: source->stateVariables())
		{
			// Skip constant variables — they are inlined at usage sites
			if (stateVar->isConstant())
				continue;
			if (stateVar->immutable())
			{
				std::string declarationId = std::to_string(stateVar->id());
				if (exportedImmutableIds.insert(declarationId).second)
					immutableDeclarations.push_back(stateVar);
				continue;
			}
			// EIP-1153 `transient`-location state variable (solc 0.8.28+,
			// `uint256 transient x;`). v1 accepts ONLY variables that
			// occupy a full 32-byte slot by themselves (value type,
			// storageBytes == 32, layout offset 0): for exactly those,
			// solc's read is the raw `tload(slot)` and the write the raw
			// `tstore(slot, value)` with identity cleanup, so the
			// runtime's raw-slot TransientStorage model
			// (Mapping U256 U256) is EXACT. Anything narrower is PACKED
			// (masked read-modify-write over a shared slot) — modeling
			// that as a raw word would be unsound precisely because a
			// reentrant callee can `tstore` garbage into the shared slot,
			// making the raw-word model and the masked read disagree —
			// so every other shape keeps a loud whole-contract refusal
			// (same fail-closed idiom as before this feature; see
			// plans/spechunt-feature-work-design.md §3.2). Public
			// visibility is refused too: the auto-getter scaffolds are
			// keyed on Storage fields and are not wired for transient.
			if (stateVar->referenceLocation() == VariableDeclaration::Location::Transient)
			{
				Type const* type = stateVar->annotation().type;
				if (!type || !type->isValueType() || type->storageBytes() != 32)
					throw UnsupportedSolCore(
						"`transient`-location state variable '" + stateVar->name()
						+ "' is not a full-slot value type; packed/narrow/reference "
						  "transient variables have no faithful raw-slot lowering "
						  "(a reentrant callee can tstore into the shared slot, so "
						  "a raw-word model of a masked read would be a false model).");
				if (stateVar->isPublic())
					throw UnsupportedSolCore(
						"public `transient`-location state variable '" + stateVar->name()
						+ "' is not supported (auto-getter scaffolding is keyed on "
						  "persistent Storage fields).");
				auto slotOffset = lookupTransientSlotOffset(_compilerStack, contract, stateVar->name());
				if (!slotOffset)
					throw UnsupportedSolCore(
						"`transient`-location state variable '" + stateVar->name()
						+ "' has no entry in solc's transientStorageLayout; refusing "
						  "to guess a slot.");
				if (slotOffset->second != 0)
					throw UnsupportedSolCore(
						"`transient`-location state variable '" + stateVar->name()
						+ "' sits at a nonzero intra-slot offset; only sole-occupant "
						  "full-slot transient variables are modeled.");
				if (exportedFieldNames.count(stateVar->name()))
					continue;
				Json field = Json::object();
				field["name"] = stateVar->name();
				field["type"] = exportTypeName(stateVar->typeName());
				field["slot"] = slotOffset->first;
				transientFields.emplace_back(std::move(field));
				exportedFieldNames.insert(stateVar->name());
				continue;
			}
			// Skip already-exported fields (can happen with diamond inheritance)
			if (exportedFieldNames.count(stateVar->name()))
				continue;
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
	for (auto const& getter: namespacedGetters)
	{
		for (auto const& member: getter.structDef->members())
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
	callEnvFields.emplace_back(Json{{"name", "origin"}, {"type", "address"}});
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
	// msg.sig: the 4-byte function selector of the current call frame,
	// modeled as a right-aligned numeric u256 (same convention as
	// `.selector` / `type(I).interfaceId`, NOT the left-shifted bytes4
	// word). Ordered-field contract: msgSig comes after calldata, and the
	// blockHashes/blobHashes pair below must stay LAST, matching the
	// frontend's ordered-field canonical-CallEnv acceptance
	// (LeanSupport.ml is_canonical_call_env).
	callEnvFields.emplace_back(Json{{"name", "msgSig"}, {"type", "u256"}});
	// BLOCKHASH / EIP-4844 BLOBHASH environment data: block-number->hash and
	// blob-index->hash mappings, unconstrained env DATA (a strict
	// over-approximation; the 256-block window/zero rule and the
	// per-transaction blob list are properties of the environment a frame
	// runs against, never rules the model states). The generator lowers the
	// `blockhash(n)` / `blobhash(i)` builtins (exported as plain
	// internal_call nodes) to lookups in these per-frame mappings; a global
	// opaque function would instead equate reads across frames with
	// different environments.
	{
		Json hashMapType = Json::object();
		hashMapType["kind"] = "mapping";
		hashMapType["key"] = "u256";
		hashMapType["value"] = "u256";
		callEnvFields.emplace_back(Json{{"name", "blockHashes"}, {"type", hashMapType}});
		callEnvFields.emplace_back(Json{{"name", "blobHashes"}, {"type", hashMapType}});
	}
	typeDecls.emplace_back(runtimeTypeDecl("CallEnv", std::move(callEnvFields)));
	Json worldStateFields = Json::array();
	worldStateFields.emplace_back(
		Json{{"name", "balances"}, {"type", Json{{"kind", "mapping"}, {"key", "address"}, {"value", "u256"}}}});
	worldStateFields.emplace_back(
		Json{{"name", "codeSize"}, {"type", Json{{"kind", "mapping"}, {"key", "address"}, {"value", "u256"}}}});
	worldStateFields.emplace_back(
		Json{
			{"name", "contractStorage"},
			{"type",
			 Json{
				 {"kind", "mapping"},
				 {"key", "address"},
				 {"value", Json{{"kind", "mapping"}, {"key", "u256"}, {"value", "u256"}}}}}});
	typeDecls.emplace_back(runtimeTypeDecl("WorldState", std::move(worldStateFields)));
	typeDecls.emplace_back(runtimeTypeDecl("Memory", Json::array()));
	typeDecls.emplace_back(runtimeTypeDecl("MemoryView", Json::array()));
	typeDecls.emplace_back(runtimeTypeDecl("ByteArray", Json::array()));
	typeDecls.emplace_back(runtimeTypeDecl("Logs", Json::array()));

	// Export user-defined struct types
	// Check top-level structs in the source unit
	for (auto const& node: contract.sourceUnit().nodes())
	{
		if (auto const* structDef = dynamic_cast<StructDefinition const*>(node.get()))
		{
			Json structFields = Json::array();
			for (auto const& member: structDef->members())
			{
				try
				{
					structFields.emplace_back(exportStructMemberField(*structDef, *member));
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
			typeDecls.emplace_back(runtimeTypeDecl(exportedStructName(*structDef), std::move(structFields)));
		}
	}
	// Also check structs defined inside contracts and interfaces
	for (auto const& node: contract.sourceUnit().nodes())
	{
		if (auto const* contractNode = dynamic_cast<ContractDefinition const*>(node.get()))
		{
			for (auto const* structDef: contractNode->definedStructs())
			{
				// Avoid duplicates (struct might already be added from source unit level)
				bool exists = false;
				for (auto const& existing: typeDecls)
					if (existing.value("name", "") == exportedStructName(*structDef))
						exists = true;
				if (!exists)
				{
					Json structFields = Json::array();
					for (auto const& member: structDef->members())
					{
						try
						{
							structFields.emplace_back(exportStructMemberField(*structDef, *member));
						}
						catch (UnsupportedSolCore const&)
						{
							// Skip struct fields with unsupported types
						}
					}
					typeDecls.emplace_back(runtimeTypeDecl(exportedStructName(*structDef), std::move(structFields)));
				}
			}
		}
	}

	// Also export struct types from all reachable source units — including
	// base contracts, imported libraries, and transitively imported files.
	{
		std::set<std::string> addedTypeNames;
		for (auto const& existing: typeDecls)
			addedTypeNames.insert(existing.value("name", ""));

		auto tryAddStruct = [&](StructDefinition const* structDef)
		{
			if (addedTypeNames.count(exportedStructName(*structDef)))
				return;
			Json structFields = Json::array();
			for (auto const& member: structDef->members())
			{
				try
				{
					structFields.emplace_back(exportStructMemberField(*structDef, *member));
				}
				catch (...)
				{
				}
			}
			typeDecls.emplace_back(runtimeTypeDecl(exportedStructName(*structDef), std::move(structFields)));
			addedTypeNames.insert(exportedStructName(*structDef));
		};

		// Collect all reachable source units by walking the import graph
		std::set<SourceUnit const*> visitedUnits;
		std::vector<SourceUnit const*> unitQueue;
		unitQueue.push_back(&contract.sourceUnit());
		for (auto const* baseContract: contract.annotation().linearizedBaseContracts)
			unitQueue.push_back(&baseContract->sourceUnit());

		while (!unitQueue.empty())
		{
			SourceUnit const* unit = unitQueue.back();
			unitQueue.pop_back();
			if (!visitedUnits.insert(unit).second)
				continue;

			// Scan this source unit for struct definitions
			for (auto const& node: unit->nodes())
			{
				if (auto const* structDef = dynamic_cast<StructDefinition const*>(node.get()))
					tryAddStruct(structDef);
				if (auto const* contractNode = dynamic_cast<ContractDefinition const*>(node.get()))
					for (auto const* structDef: contractNode->definedStructs())
						tryAddStruct(structDef);
				// Follow imports to reach transitively imported source units
				if (auto const* importDir = dynamic_cast<ImportDirective const*>(node.get()))
					if (importDir->annotation().sourceUnit)
						unitQueue.push_back(importDir->annotation().sourceUnit);
			}
		}
	}

	solcore["type_decls"] = std::move(typeDecls);
	// EIP-1153 transient state variables (see exportStateVars): additive
	// crate field; absent/empty means what it always meant (no transient
	// state). Old frontends that predate the field never read it, and the
	// transient_get/transient_set nodes referencing these names fail closed
	// there via the unknown-kind Parse_error path.
	bool const hasTransientState = !transientFields.empty();
	if (hasTransientState)
		solcore["transient_state"] = std::move(transientFields);
	solcore["sub_storage_getters"] = std::move(subStorageGetters);
	Json foreignContracts = Json::array();
	// exportForeignContractSummary is a pure function of (compilerStack,
	// candidate). Cache one complete summary per candidate for the stack run;
	// required ABI metadata failures propagate and reject the export rather
	// than silently deleting a contract or method.
	static thread_local CompilerStack const* cachedSummaryStack = nullptr;
	static thread_local std::map<std::string, Json> foreignSummaryCache;
	if (cachedSummaryStack != &_compilerStack)
	{
		foreignSummaryCache.clear();
		cachedSummaryStack = &_compilerStack;
	}
	for (std::string const& candidate: _compilerStack.contractNames())
	{
		auto it = foreignSummaryCache.find(candidate);
		if (it == foreignSummaryCache.end())
			it = foreignSummaryCache.emplace(
				candidate, exportForeignContractSummary(_compilerStack, candidate)).first;
		foreignContracts.emplace_back(it->second);
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
	//
	// EXCEPTION (transient state-variable feature): when this contract
	// declares `transient`-location STATE VARIABLES, the generator's
	// high-level transient_get/transient_set lowering projects/updates the
	// state's `transient` component through the ordinary record-field
	// machinery, which requires the field to exist in `state.fields`. The
	// type is spelled as the raw `mapping u256 -> u256` (matching the
	// runtime's `TransientStorage := Mapping U256 U256`), NOT the named
	// "TransientStorage" spelling that broke real-corpus generation (see
	// above) — the mapping form goes through the synthetic mapping-type
	// registration every storage mapping field already uses. The field is
	// appended LAST, mirroring the runtime ExecState's field order, and
	// only when transient variables exist, so every other contract keeps
	// the exact 6-field legacy shape.
	if (hasTransientState)
		stateFields.emplace_back(
			Json{{"name", "transient"}, {"type", Json{{"kind", "mapping"}, {"key", "u256"}, {"value", "u256"}}}});
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
	for (auto const& getter: namespacedGetters)
		namespacedGetterDefinitions.insert(getter.function);

	std::vector<FunctionDefinition const*> overloadCandidates;
	for (FunctionDefinition const* function: contract.definedFunctions())
	{
		if (!function->isOrdinary() || !function->isImplemented())
			continue;
		if (namespacedGetterDefinitions.count(function))
			continue;
		overloadCandidates.push_back(function);
	}
	for (auto const* baseContract: contract.annotation().linearizedBaseContracts)
	{
		if (baseContract == &contract)
			continue;
		for (FunctionDefinition const* function: baseContract->definedFunctions())
		{
			if (!function->isOrdinary() || !function->isImplemented())
				continue;
			if (namespacedGetterDefinitions.count(function))
				continue;
			overloadCandidates.push_back(function);
		}
	}
	// Bucket-1 exporter closure (spechunt-reject-buckets-design §1.2.2):
	// insert every out-of-linearization library/free-function callee into the
	// overload-candidate set BEFORE assignExportedFunctionNames runs, so call
	// sites and the pass-4 definitions below bake IDENTICAL names, and any
	// cross-scope overload is disambiguated consistently at both ends. The
	// generator resolves internal calls BY NAME ONLY, so this ordering is
	// load-bearing: pre-assignment propagates to every call-site path through
	// the exportedFunctionNames memo consulted first by exportedFunctionName.
	std::vector<FunctionDefinition const*> outOfLinearizationCallees = collectOutOfLinearizationCallees(contract);
	{
		std::set<FunctionDefinition const*>
			linearizationCandidates(overloadCandidates.begin(), overloadCandidates.end());
		for (FunctionDefinition const* function: outOfLinearizationCallees)
			if (!linearizationCandidates.count(function))
				overloadCandidates.push_back(function);
	}
	assignExportedFunctionNames(overloadCandidates);
	// Collision rule (§1.2.2, extended by spechunt-remaining-rejects-design
	// §B1): if after typed-suffix disambiguation a collected
	// out-of-linearization callee still shares its exported name with ANY
	// other candidate (a linearization function, or another collected callee
	// from a different scope), that is a silent name-only mis-bind
	// downstream — the generator binds internal calls BY NAME ONLY
	// (Analysis.ml validate_internal_call_catalog over one flat
	// constructors @ internal_functions @ functions namespace), so a harness
	// wrapper `average(uint256,uint256)` around `Math.average(uint256,uint256)`
	// would let the wrapper's body call ITSELF. Same-name sharing WITHIN a
	// linearization is legitimate (virtual override chains deliberately share
	// their slot name), but a library/free function never participates in
	// virtual dispatch, so any share involving one is a defect.
	//
	// ONE round of scope-qualified renaming is applied to the COLLECTED side
	// of each such group, followed by a fail-closed recheck. Never
	// qualify-and-continue: if any group still contains a collected callee
	// after qualification, the original loud throw stands.
	//
	// Why this is provably collision-free at BOTH the call site and the
	// definition:
	//   (a) every call-site emission path and the pass-4 definition emission
	//       derive the name from the SAME `exportedFunctionNames` memo entry
	//       (exportedFunctionName reads the memo first, :2380-2384), and this
	//       block runs before ANY body export (the constructor at :10500 is
	//       the first) — so both ends are assigned once, together;
	//   (b) the recheck below runs over the COMPLETE candidate set
	//       (linearization + collected), so no exported internal function can
	//       share a final name with a collected callee;
	//   (c) the pass-4 rename tripwire further below throws if
	//       `disambiguate()` renames an embedded entry, so the JSON-level
	//       pass cannot silently undo (a).
	// Only the COLLECTED member of a colliding group is ever renamed:
	// renaming a linearization member would desync dispatch entries, super
	// aliases and virtual-slot names, all of which are keyed on the plain
	// slot name.
	{
		std::set<FunctionDefinition const*>
			collectedSet(outOfLinearizationCallees.begin(), outOfLinearizationCallees.end());
		auto groupByExportedName = [&]()
		{
			std::map<std::string, std::vector<FunctionDefinition const*>> byExportedName;
			for (FunctionDefinition const* function: overloadCandidates)
				byExportedName[exportedFunctionName(*function)].push_back(function);
			return byExportedName;
		};
		// Round 1 — qualify. Note this fires ONLY for groups that would have
		// thrown before this change, so every previously-exportable contract
		// keeps byte-identical exported names.
		for (auto const& [exportedName, group]: groupByExportedName())
		{
			if (group.size() <= 1)
				continue;
			for (FunctionDefinition const* function: group)
			{
				if (!collectedSet.count(function))
					continue;
				if (auto qualified = scopeQualifiedCalleeName(*function, exportedName))
					exportedFunctionNames[function] = *qualified;
			}
		}
		// Round 2 — fail-closed recheck over the complete candidate set. No
		// second qualification round: an unresolved collision here (a user
		// function literally named `average__lib__Math`, two same-stem source
		// files, or a callee shape with no qualification arm) is refused, not
		// renamed again.
		for (auto const& [exportedName, group]: groupByExportedName())
		{
			if (group.size() <= 1)
				continue;
			for (FunctionDefinition const* function: group)
				if (collectedSet.count(function))
					throw UnsupportedSolCore(
						"exported internal-function name '" + exportedName
						+ "' is shared between an out-of-linearization library/free-function "
						  "callee and another function, and scope qualification did not "
						  "separate them; the generator binds internal calls by "
						  "name only, so emitting both would silently mis-bind call sites. "
						  "Refusing to export this contract.");
		}
	}
	std::set<FunctionDefinition const*> superReferencedFunctions = collectSuperReferencedFunctions(contract);
	for (FunctionDefinition const* function: superReferencedFunctions)
	{
		std::string alias = contractScopedSuperAlias(*function);
		contractScopedSuperAliases[function] = alias;
		exportedFunctionNames[function] = std::move(alias);
	}

	// Ordering constraint (load-bearing): event names must be assigned BEFORE
	// any body export, because every `emit` site reads the memo. Body export
	// begins with the constructor immediately below.
	//
	// The assigned set is the INHERITED closure (`interfaceEvents`), not just
	// the contract's own `events()`: a derived contract emits events declared
	// on its bases (and used library events), and the consumer's structured
	// event model (spec-hunt Class D2) refuses an emit with no matching
	// declaration rather than fabricating a log shape — so every emittable
	// event must be declared in this contract's export. Running the
	// disambiguation over the closure also catches cross-contract overloads
	// (base `E(uint256)` + derived `E(address)`) that per-contract assignment
	// could silently merge. `false`: no call-graph requirement, so abstract
	// units (which export too) keep working; when the call graph IS set the
	// closure additionally picks up emitted library events.
	assignExportedEventNames(contract.interfaceEvents(false));

	Json immutables = Json::array();
	for (VariableDeclaration const* immutable: immutableDeclarations)
	{
		Json entry = Json::object();
		entry["declarationId"] = std::to_string(immutable->id());
		entry["name"] = immutable->name();
		entry["type"] = exportTypeName(immutable->typeName());
		entry["abi"] = exportAbiDescriptor(immutable->name(), immutable->annotation().type, false);
		entry["sourceLocation"] = sourceLocation(immutable->location());
		entry["initializer"] = immutable->value() ? exportExpr(*immutable->value()) : Json();
		immutables.emplace_back(std::move(entry));
	}
	solcore["immutables"] = std::move(immutables);

	Json constructorDisposition = exportConstructorDisposition(contract);
	if (auto const* constructor = contract.constructor())
		solcore["constructor"] = exportConstructor(*constructor, contract);
	else if (inheritedConstructorHasWork(contract))
		solcore["constructor"] = exportImplicitConstructor(contract);
	solcore["constructorDisposition"] = std::move(constructorDisposition);

	Json dispatchEntries = Json::array();
	for (auto const& [selector, functionType]: contract.interfaceFunctions())
	{
		(void) selector;
		if (!functionType)
			throw UnsupportedSolCore(
				"Contract '" + contract.name()
				+ "' has a dispatch entry without a compiler-resolved function type.");
		dispatchEntries.emplace_back(exportDispatchEntry(
			functionType, dynamic_cast<FunctionDefinition const*>(&functionType->declaration())));
	}
	solcore["dispatch_entries"] = std::move(dispatchEntries);

	Json functions = Json::array();
	Json internalFunctions = Json::array();

	// Collect exported internal function names to avoid duplicates
	std::set<std::string> exportedInternalNames;

	subStorageGetters = Json::array();
	for (auto const& getter: namespacedGetters)
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
			functions.emplace_back(exportFunction(*function, contract));
			if (isPublicLibraryStructuralStorageFunction(*function))
				internalFunctions.emplace_back(exportFunction(
					*function,
					contract,
					/*_isInternal=*/true,
					/*_spec=*/nullptr,
					/*_storageRefCompanion=*/true));
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
	for (auto const& f: internalFunctions)
		if (f.contains("name") && f["name"].is_string())
			exportedInternalNames.insert(f["name"].get<std::string>());

	// Track which public function names we already exported
	std::set<std::string> exportedPublicNames;
	for (auto const& f: functions)
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
			for (FunctionDefinition const* function: baseContract->definedFunctions())
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
				functions.emplace_back(exportFunction(*function, contract));
				exportedPublicNames.insert(exportedFunctionName(*function));
			}
		}
	}

	// Third pass: export inherited internal functions from base contracts
	// (most-derived first so overrides win)
	for (auto const* baseContract: contract.annotation().linearizedBaseContracts)
	{
		if (baseContract == &contract)
			continue; // Skip self
		for (FunctionDefinition const* function: baseContract->definedFunctions())
		{
			if (!function->isOrdinary() || !function->isImplemented())
				continue;
			// Skip namespaced storage getter functions
			if (namespacedGetterDefinitions.count(function))
				continue;
			bool needsSuperAlias = superReferencedFunctions.count(function);
			if ((function->visibility() == Visibility::Public || function->visibility() == Visibility::External)
				&& !needsSuperAlias)
				continue; // Already handled in public pass above
			if (exportedInternalNames.count(exportedFunctionName(*function)))
				continue; // Already exported (overridden in derived contract)
			try
			{
				internalFunctions.emplace_back(exportFunction(*function, contract, /*_isInternal=*/true));
				exportedInternalNames.insert(exportedFunctionName(*function));
			}
			catch (UnsupportedSolCore const&)
			{
			}
			catch (std::exception const&)
			{
			}
		}
	}

	auto exportAbiSpecialFunction = [&](FunctionDefinition const& _function, std::string const& _name)
	{
		Json f = Json::object();
		f["name"] = _name;
		f["params"] = Json::array();
		for (auto const& parameter: _function.parameters())
			f["params"].emplace_back(exportParam(*parameter, true, false));
		f["returnAbi"] = Json::array();
		for (auto const& parameter: _function.returnParameters())
			f["returnAbi"].emplace_back(exportAbiDescriptor(
				parameter->name(), parameter->annotation().type, false));
		if (_function.returnParameters().empty())
			f["return"] = Json("unit");
		else if (_function.returnParameters().size() == 1)
			f["return"] = exportTypeName(_function.returnParameters().front()->typeName());
		else
			throw UnsupportedSolCore(
				"ABI-visible " + _name + " function has more than one return parameter.");
		f["body"] = exportBody(_function);
		f["ast_write_oracle"] = astWriteOracleJson(_function, contract);
		return f;
	};

	if (auto const* recv = contract.receiveFunction())
		if (recv->isImplemented())
			functions.emplace_back(exportAbiSpecialFunction(*recv, "receive"));
	if (auto const* fb = contract.fallbackFunction())
		if (fb->isImplemented())
			functions.emplace_back(exportAbiSpecialFunction(*fb, "fallback"));

	// Fourth pass (bucket-1 exporter closure, §1.2.3): emit every collected
	// out-of-linearization library/free-function definition into this crate's
	// internal_functions, so the call sites exported by the passes above
	// finally have their definitions in the same unit. Placed BEFORE the
	// fn-ptr specialization drain below so specializations discovered inside
	// these bodies still drain. The per-function try/catch keeps the
	// fail-closed skip semantics of passes 1-3: if a library/free body is
	// inexportable, the definition is skipped, the call site dangles, and the
	// generator's catalog gate rejects exactly as today.
	std::vector<std::pair<size_t, std::string>> outOfLinearizationEntryNames;
	for (FunctionDefinition const* function: outOfLinearizationCallees)
	{
		std::string calleeName = exportedFunctionName(*function);
		if (exportedInternalNames.count(calleeName))
			continue;
		try
		{
			internalFunctions.emplace_back(exportFunction(*function, contract, /*_isInternal=*/true));
			outOfLinearizationEntryNames.emplace_back(internalFunctions.size() - 1, calleeName);
			exportedInternalNames.insert(calleeName);
		}
		catch (UnsupportedSolCore const&)
		{
			// Skip unsupported out-of-linearization callees — the dangling
			// call site keeps today's fail-closed generator rejection.
		}
		catch (std::exception const&)
		{
			// Skip callees that cause unexpected export errors.
		}
	}

	// --- Emit specialized siblings discovered while lowering the passes
	// above (bounded defunctionalization — see the block preceding
	// exportExpr's definition). Runs to a fixpoint: exporting a specialized
	// sibling's body can itself enqueue further specializations (transitive
	// forwarding, e.g. `f(op)` calling `g(op)`). Termination:
	// fnPtrSpecializationsByName memoizes by name, and the
	// (callee × binding-tuple) space is finite (drawn from the linearized
	// hierarchy's own FunctionDefinitions), so a self-recursive
	// `f(op){ ... f(op) ... }` re-derives the SAME specialized name on its
	// own recursive call (memo hit) — ordinary self-recursion in the
	// emitted body, not a new request.
	//
	// Deliberately NOT wrapped in try/catch per request (unlike every other
	// pass above): signature emission cannot throw (param/return export
	// already falls back to u256 on any failure) and body-export failures
	// are absorbed by exportBody's own catch-all into "unsupported_body" —
	// so every queued name is guaranteed to land in internal_functions and
	// callers never dangle on a missing callee. If exportFunction were ever
	// to throw here regardless (exporter drift), propagating it up and
	// failing the WHOLE contract export loudly is the fail-closed choice —
	// strictly safer than silently dropping a callee a caller's body
	// already references by name.
	while (!fnPtrSpecializationQueue.empty())
	{
		std::string specializedName = fnPtrSpecializationQueue.front();
		fnPtrSpecializationQueue.pop_front();
		// Each name is enqueued exactly once (registerFnPtrSpecialization
		// enqueues only on first registration) and nothing before this drain
		// emits specializations — so a hit here can only mean a USER-DEFINED
		// internal function is literally named like this specialization.
		// Skipping emission would silently bind every rewritten call site to
		// that unrelated same-named function (a silent mis-bind, the exact
		// fail-open class this feature closes); merging is equally wrong.
		// Fail the whole export loudly instead.
		if (exportedInternalNames.count(specializedName))
			throw UnsupportedSolCore(
				"internal-function-value specialization name '" + specializedName
				+ "' collides with an already-exported internal function of the same name; "
				  "refusing to bind rewritten call sites to an unrelated definition.");
		auto requestIt = fnPtrSpecializationsByName.find(specializedName);
		if (requestIt == fnPtrSpecializationsByName.end())
			continue; // Unreachable in practice: every queued name was just registered alongside its request.
		FnPtrSpecializationRequest const& request = requestIt->second;
		internalFunctions.emplace_back(exportFunction(*request.callee, contract, /*_isInternal=*/true, &request));
		exportedInternalNames.insert(specializedName);
	}

	// Disambiguate overloaded function names.
	// Lean doesn't support overloading, so functions with the same name
	// get a suffix based on parameter count or parameter types.
	auto disambiguate = [](Json& funcs)
	{
		// Count occurrences of each name
		std::map<std::string, int> nameCounts;
		for (auto const& f: funcs)
			if (f.contains("name") && f["name"].is_string())
				nameCounts[f["name"].get<std::string>()]++;

		// For names that appear more than once, disambiguate
		std::map<std::string, int> nameIndex;
		for (auto& f: funcs)
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
					for (auto const& p: f["params"])
					{
						if (!typeSuffix.empty())
							typeSuffix += "_";
						std::string ty = "u256";
						if (p.contains("type"))
						{
							if (p["type"].is_string())
								ty = p["type"].get<std::string>();
							else if (p["type"].is_object() && p["type"].contains("kind"))
								ty = p["type"]["kind"].get<std::string>();
						}
						// Abbreviate common types
						if (ty == "address")
							ty = "addr";
						else if (ty == "mapping")
							ty = "map";
						else if (ty.length() > 4)
							ty = ty.substr(0, 4);
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

	// §1.2 defense: names for the bucket-1 embedded callees were pre-assigned
	// uniquely (collision rule above), so disambiguate() must have been a
	// no-op for them — a rename here would desync every call site baked with
	// the pre-assigned name. Fail loudly instead of shipping a mis-bind.
	for (auto const& [entryIndex, expectedName]: outOfLinearizationEntryNames)
	{
		Json const& entry = internalFunctions.at(entryIndex);
		if (!entry.contains("name") || !entry["name"].is_string() || entry["name"].get<std::string>() != expectedName)
			throw UnsupportedSolCore(
				"embedded out-of-linearization callee '" + expectedName
				+ "' was renamed by overload disambiguation; call sites would "
				  "silently desync. Refusing to export this contract.");
	}

	// Freeze the dynamically discovered mint sites into deterministic,
	// producer-proven closed candidate tables. Assembly that references a
	// typed value containing an internal-function pointer can manufacture a
	// candidate outside that table. Persistent values are contract-wide:
	// arbitrary assembly can address their storage slot without naming the
	// Solidity declaration. Aggregate memory values are conservative too,
	// because raw mstore can bypass the compiler's external-reference map.
	//
	// The poisoning is function-granular, not contract-granular: only a
	// function whose export involves an internal-function VALUE (a typed
	// internal_function node, or a call routed through a candidate table's
	// dispatcher/delegate) depends on the closed-table guarantee — direct
	// calls never do. Demote exactly those functions to unsupported_body
	// (carrying this refusal reason) and drop the voided tables, so the
	// rest of the contract — e.g. OZ Arrays' unsafeAccess family beside its
	// open-world fn-pointer quicksort — keeps its exact model.
	if (!internalFnTablesByFingerprint.empty()
		&& (internalFnAssemblyTouchesValue
			|| (internalFnContractContainsAssembly
				&& (contractHasStoredInternalFnValue(contract)
					|| contractHasAggregateInternalFnValue(contract)))))
	{
		std::string const poisonReason =
			"Inline assembly can mutate an internal-function value outside its closed candidate table; "
			"raw assembly code-pointer construction is open-world, so every function whose export "
			"involves an internal-function value is refused.";
		std::set<std::string> tableNames;
		for (auto const& [fingerprint, table]: internalFnTablesByFingerprint)
		{
			(void) fingerprint;
			tableNames.insert(table.dispatcherName);
			for (auto const& [tag, candidate]: table.candidatesByTag)
			{
				(void) tag;
				tableNames.insert(candidate.delegateName);
			}
		}
		std::function<bool(Json const&)> usesInternalFnValue = [&](Json const& _node) -> bool
		{
			if (_node.is_object())
			{
				auto kind = _node.find("kind");
				if (kind != _node.end() && kind->is_string() && kind->get<std::string>() == "internal_function")
					return true;
				for (auto const& [key, value]: _node.items())
				{
					(void) key;
					if (usesInternalFnValue(value))
						return true;
				}
			}
			else if (_node.is_array())
			{
				for (auto const& value: _node)
					if (usesInternalFnValue(value))
						return true;
			}
			else if (_node.is_string() && tableNames.count(_node.get<std::string>()))
				return true;
			return false;
		};
		auto demote = [&](Json& _fn)
		{
			if (!usesInternalFnValue(_fn))
				return;
			Json failedBody = Json::object();
			failedBody["kind"] = "unsupported_body";
			failedBody["error"] = poisonReason;
			_fn["body"] = std::move(failedBody);
		};
		for (auto& fn: functions)
			demote(fn);
		for (auto& fn: internalFunctions)
			demote(fn);
		if (solcore.contains("constructor"))
			demote(solcore["constructor"]);
		internalFnTablesByFingerprint.clear();
	}
	auto functionNameExists = [&](std::string const& _name)
	{
		for (Json const* entries: {&functions, &internalFunctions})
			for (auto const& entry: *entries)
				if (entry.contains("name") && entry["name"].is_string() && entry["name"].get<std::string>() == _name)
					return true;
		return false;
	};
	std::set<std::string> tableIds;
	std::set<std::string> generatedNames;
	Json internalFnTables = Json::array();
	for (auto const& [fingerprint, table]: internalFnTablesByFingerprint)
	{
		(void) fingerprint;
		if (!tableIds.insert(table.tableId).second)
			throw UnsupportedSolCore(
				"Two distinct internal-function types received candidate table id '" + table.tableId + "'.");
		if (!generatedNames.insert(table.dispatcherName).second || functionNameExists(table.dispatcherName))
			throw UnsupportedSolCore("Internal-function dispatcher name collision at '" + table.dispatcherName + "'.");
		if (!table.fnType.is_object() || table.fnType.value("kind", std::string{}) != "internal_function")
			throw UnsupportedSolCore("Internal-function candidate table lost its exact function type.");
		Json tableJson = Json::object();
		tableJson["id"] = table.tableId;
		tableJson["type"] = table.fnType;
		tableJson["dispatcher"] = table.dispatcherName;
		tableJson["candidates"] = Json::array();
		for (auto const& [tag, candidate]: table.candidatesByTag)
		{
			if (tag == "0x0000000000000000")
				throw UnsupportedSolCore("Internal-function candidate attempted to use reserved zero tag.");
			if (!functionNameExists(candidate.functionName))
				throw UnsupportedSolCore(
					"Internal-function candidate '" + candidate.identity
					+ "' is outside this artifact's closed function set.");
			if (!generatedNames.insert(candidate.delegateName).second || functionNameExists(candidate.delegateName))
				throw UnsupportedSolCore(
					"Internal-function delegate name collision at '" + candidate.delegateName + "'.");
			Json candidateJson = Json::object();
			candidateJson["tag"] = tag;
			candidateJson["function"] = candidate.functionName;
			candidateJson["declarationContractId"] = candidate.declarationContractId;
			candidateJson["targetContractId"] = candidate.targetContractId;
			candidateJson["delegate"] = candidate.delegateName;
			tableJson["candidates"].emplace_back(std::move(candidateJson));
		}
		internalFnTables.emplace_back(std::move(tableJson));
	}
	solcore["internal_fn_tables"] = std::move(internalFnTables);

	solcore["functions"] = std::move(functions);
	solcore["internal_functions"] = std::move(internalFunctions);

	// Additive schema field (SOLCORE_UNRESOLVED_INTERNAL_CALL_DESIGN):
	// exact set of virtual-slot callee names that resolved to no
	// implementation anywhere in this unit's own inheritance linearization
	// (populated only while exporting an abstract/interface unit — see
	// `virtualCallTargetName`). Always emitted, even when empty, so
	// consumers can distinguish "no unbound slots" from "field absent /
	// artifact predates this signal".
	Json unboundVirtualSlots = Json::array();
	for (std::string const& slotName: unboundVirtualSlotNames)
		unboundVirtualSlots.push_back(slotName);
	solcore["unbound_virtual_slots"] = std::move(unboundVirtualSlots);

	// Same INHERITED closure as `assignExportedEventNames` above — the two
	// must iterate the identical set, or an emit site could bind (via the
	// memo) to a name whose declaration was never exported.
	Json events = Json::array();
	for (EventDefinition const* event: contract.interfaceEvents(false))
	{
		if (!event)
			throw UnsupportedSolCore(
				"Compiler-resolved event declaration closure contains a null declaration.");
		events.emplace_back(exportEvent(*event));
	}
	solcore["events"] = std::move(events);

	// Export enum declarations from every source unit reachable through the
	// active compiler stack — not just the contract's own SourceUnit. Enums
	// defined in imported library files (e.g. RoundingMode in Fixed.sol) must
	// be available so downstream consumers can resolve `EnumName.VARIANT`
	// constants to their integer values. (REV-271/bug_013 fix.)
	Json enumDecls = Json::array();
	auto recordEnum = [&](EnumDefinition const& _enumDef)
	{
		auto const enumName = exportedEnumName(_enumDef);
		// Avoid exact duplicates while preserving distinct same-named enums
		// from different source units under qualified names.
		for (auto const& existing: enumDecls)
			if (existing.value("name", "") == enumName)
				return;
		Json decl = Json::object();
		decl["name"] = enumName;
		decl["variants"] = Json::array();
		for (auto const& member: _enumDef.members())
			decl["variants"].emplace_back(member->name());
		enumDecls.emplace_back(std::move(decl));
	};
	auto walkSourceUnit = [&](SourceUnit const& _unit) { forEachEnumDefinition(_unit, recordEnum); };
	if (activeCompilerStack)
	{
		for (auto const& sourceName: activeCompilerStack->sourceNames())
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

	// Producer capability stamp: every legacy artifact identifies the exact
	// measured row-set version. Consumers reject legacy artifacts where this
	// stamp is absent or differs, and separately record it only when a
	// codegen-divergent row is used.
	if (!_compilerStack.viaIR())
		solcore["evalOrderCommitment"] = EvalOrderCommitment;
	Json origins = exportOrigins(contract);
	for (auto const& [key, value]: metadata.items())
		origins[key] = value;

	return {std::move(solcore), std::move(origins)};
}

}

} // namespace

solidity::frontend::solcore::ExportArtifacts
solidity::frontend::solcore::exportContract(CompilerStack const& _compilerStack, std::string const& _contractName)
{
	try
	{
		if (VersionCompactBytes.size() >= 2 && VersionCompactBytes[0] == 0 && VersionCompactBytes[1] == 8)
			return v0_8::exportContract(_compilerStack, _contractName);
	}
	catch (UnsupportedSolCore const& exception)
	{
		return {
			unsupportedExport(_contractName, exception.what(), _compilerStack.viaIR()),
			unsupportedExport(_contractName, exception.what(), _compilerStack.viaIR())};
	}
	catch (std::exception const& exception)
	{
		// Catch any other exception that escaped (e.g., from nlohmann::json, dynamic_cast, etc.)
		std::string reason = "Internal error: "s + exception.what();
		return {
			unsupportedExport(_contractName, reason, _compilerStack.viaIR()),
			unsupportedExport(_contractName, reason, _compilerStack.viaIR())};
	}
	catch (...)
	{
		// Catch truly unknown exceptions
		std::string reason = "Internal error: unknown exception during SolCore export";
		return {
			unsupportedExport(_contractName, reason, _compilerStack.viaIR()),
			unsupportedExport(_contractName, reason, _compilerStack.viaIR())};
	}

	std::string reason = "No SolCore exporter is registered for compiler version " + VersionString + ".";
	return {
		unsupportedExport(_contractName, reason, _compilerStack.viaIR()),
		unsupportedExport(_contractName, reason, _compilerStack.viaIR())};
}
