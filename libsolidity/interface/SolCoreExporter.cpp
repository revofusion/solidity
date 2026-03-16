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

#include <stdexcept>

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
	flags["constructors"] = false;
	flags["externalCalls"] = true;
	flags["multipleReturns"] = false;
	flags["structs"] = false;
	flags["enums"] = false;
	flags["inheritance"] = false;
	flags["modifiers"] = false;
	flags["inlineAssembly"] = true;
	flags["fixedBytes"] = true;
	flags["smallUints"] = true;
	flags["signedInts"] = true;
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
			Json result = Json::object();
			result["kind"] = "enum";
			result["name"] = std::string(path.back());
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
	}
	if (_base == "block")
	{
		if (_member == "timestamp")
			return "blockTimestamp";
		if (_member == "number")
			return "blockNumber";
	}
	throw UnsupportedSolCore("Unsupported magic member access: " + _base + "." + _member);
}

Json exportExpr(Expression const& _expr);

std::pair<std::vector<std::string>, Json> exportStorageMapLValue(Expression const& _expr)
{
	auto const* indexAccess = dynamic_cast<IndexAccess const*>(&_expr);
	if (!indexAccess || !indexAccess->indexExpression())
		throw UnsupportedSolCore("Expected single-level mapping index access.");

	Expression const& base = indexAccess->baseExpression();
	if (auto const* identifier = dynamic_cast<Identifier const*>(&base))
	{
		auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
		if (decl && decl->isStateVariable())
			return {{decl->name()}, exportExpr(*indexAccess->indexExpression())};
	}

	throw UnsupportedSolCore("Only direct state-mapping index access is supported.");
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
		if (literal->token() == Token::StringLiteral || literal->token() == Token::UnicodeStringLiteral || literal->token() == Token::HexStringLiteral)
		{
			Json result = Json::object();
			result["kind"] = "string";
			result["value"] = literal->value();
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
				result["enum"] = enumDef->name();
				result["variant"] = enumValue->name();
				return result;
			}
		}
		if (auto const* decl = dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration))
		{
			// For constant variables, try to inline the value
			if (decl->isConstant() && decl->value())
			{
				return exportExpr(*decl->value());
			}
			// For immutable variables with a compile-time value, try to inline the value.
			// If inlining fails (e.g., the value is a function pointer or unsupported expression),
			// fall back to storage_get so the field must be in Storage.
			if (decl->immutable() && decl->value())
			{
				try
				{
					return exportExpr(*decl->value());
				}
				catch (...)
				{
					// Fall through to storage_get below
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
		// Identifiers that reference contracts/interfaces/libraries
		// These are address-typed in expressions (e.g., address(L))
		if (dynamic_cast<ContractDefinition const*>(identifier->annotation().referencedDeclaration))
		{
			// Library/contract address — return zero address as placeholder
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = "0";
			return result;
		}
		// Identifiers that reference function definitions used as values (function pointers)
		// Internal function pointers are represented as U256 at the EVM level
		if (dynamic_cast<FunctionDefinition const*>(identifier->annotation().referencedDeclaration))
		{
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = "0";
			return result;
		}
		// Identifiers that reference user-defined value type definitions (e.g., MyAddress in MyAddress.wrap)
		if (dynamic_cast<UserDefinedValueTypeDefinition const*>(identifier->annotation().referencedDeclaration))
		{
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = "0";
			return result;
		}
		// Any other non-variable declaration used in expression context
		// (e.g., struct name, error name, event name) — emit a U256 placeholder
		// since these are type-level references, not runtime values.
		// The VariableDeclaration case was already handled above, so only
		// non-variable declarations reach here.
		if (identifier->annotation().referencedDeclaration)
		{
			Json result = Json::object();
			result["kind"] = "u256";
			result["value"] = "0";
			return result;
		}
		// Fallback: no declaration found — assume it's a local variable reference
		Json result = Json::object();
		result["kind"] = "local";
		result["name"] = identifier->name();
		return result;
	}

	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
	{
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
					result["enum"] = enumDef->name();
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
				// address(expr).balance → get contract balance from world state
				Json result = Json::object();
				result["kind"] = "state_get";
				result["path"] = jsonStringArray({"world", "contractBalance"});
				return result;
			}
		}

		// Array .length access
		if (memberAccess->memberName() == "length")
		{
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
						// Fallback for unknown type argument — return 0
						Json result = Json::object();
						result["kind"] = "u256";
						result["value"] = "0";
						return result;
					}
					if (memberAccess->memberName() == "interfaceId")
					{
						// type(SomeInterface).interfaceId — return as bytes4
						Json result = Json::object();
						result["kind"] = "u256";
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

		Json result = Json::object();
		result["kind"] = "field";
		try
		{
			result["base"] = exportExpr(memberAccess->expression());
		}
		catch (...)
		{
			// If the base expression is unsupported, use a zero placeholder
			Json zero = Json::object();
			zero["kind"] = "u256";
			zero["value"] = "0";
			result["base"] = zero;
		}
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
		switch (binary->getOperator())
		{
		case Token::Add:
			result["kind"] = "u256_add";
			break;
		case Token::Sub:
			result["kind"] = "u256_sub";
			break;
		case Token::Mul:
			result["kind"] = "u256_mul";
			break;
		case Token::Div:
			result["kind"] = "u256_div";
			break;
		case Token::Mod:
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
			result["kind"] = "u256_lt";
			break;
		case Token::LessThanOrEqual:
			result["kind"] = "u256_le";
			break;
		case Token::GreaterThan:
			result["kind"] = "u256_gt";
			break;
		case Token::GreaterThanOrEqual:
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
		// Type conversions (e.g. uint256(x), address(x)) are no-ops at EVM level
		if (*call->annotation().kind == FunctionCallKind::TypeConversion)
		{
			if (!call->arguments().empty())
				return exportExpr(*call->arguments().front());
			throw UnsupportedSolCore("Empty type conversion in SolCore exporter.");
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
			for (auto const& arg: call->arguments())
			{
				try { result["args"].emplace_back(exportExpr(*arg)); }
				catch (...) {
					Json zero = Json::object();
					zero["kind"] = "u256";
					zero["value"] = "0";
					result["args"].emplace_back(zero);
				}
			}
			return result;
		}

		// Handle new expressions: new bytes(n), new string(n), new uint256[](n)
		if (auto const* newExpr = dynamic_cast<NewExpression const*>(&call->expression()))
		{
			Json result = Json::object();
			result["kind"] = "internal_call";
			result["function"] = "new_array";
			result["args"] = Json::array();
			// The argument is the size
			for (auto const& arg: call->arguments())
			{
				try { result["args"].emplace_back(exportExpr(*arg)); }
				catch (...) {
					Json zero = Json::object();
					zero["kind"] = "u256";
					zero["value"] = "0";
					result["args"].emplace_back(zero);
				}
			}
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
				for (auto const& arg: call->arguments())
				{
					try { result["args"].emplace_back(exportExpr(*arg)); }
					catch (...) {
						Json zero = Json::object();
						zero["kind"] = "u256";
						zero["value"] = "0";
						result["args"].emplace_back(zero);
					}
				}
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

			// Internal function call: callee references a FunctionDefinition
			auto const* funcDef = dynamic_cast<FunctionDefinition const*>(callee->annotation().referencedDeclaration);
			if (funcDef)
			{
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = callee->name();
				result["args"] = Json::array();
				for (auto const& arg: call->arguments())
					result["args"].emplace_back(exportExpr(*arg));
				return result;
			}
		}

		// Check for member calls on arrays and contracts
		if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&call->expression()))
		{
			// abi.encode, abi.encodePacked, abi.decode, abi.encodeWithSelector, abi.encodeWithSignature
			{
				Type const* baseType = memberAccess->expression().annotation().type;
				if (baseType && baseType->category() == Type::Category::Magic)
				{
					auto const* magicType = dynamic_cast<MagicType const*>(baseType);
					if (magicType && magicType->kind() == MagicType::Kind::ABI)
					{
						Json result = Json::object();
						result["kind"] = "internal_call";
						result["function"] = "abi_" + memberAccess->memberName();
						result["args"] = Json::array();
						for (auto const& arg: call->arguments())
						{
							try { result["args"].emplace_back(exportExpr(*arg)); }
							catch (...) {
								Json zero = Json::object();
								zero["kind"] = "u256";
								zero["value"] = "0";
								result["args"].emplace_back(zero);
							}
						}
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

			// External contract call: contract.method(args)
			Type const* baseType = memberAccess->expression().annotation().type;
			if (baseType && baseType->category() == Type::Category::Contract)
			{
				Json result = Json::object();
				result["kind"] = "external_call";
				result["target"] = exportExpr(memberAccess->expression());
				result["method"] = memberAccess->memberName();
				result["args"] = Json::array();
				for (auto const& arg: call->arguments())
					result["args"].emplace_back(exportExpr(*arg));
				return result;
			}

			// Library-qualified or type-qualified function call: L.f(args)
			// The member access resolves to a FunctionDefinition when calling through a library/type namespace
			if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
			{
				(void)funcDef;
				Json result = Json::object();
				result["kind"] = "internal_call";
				result["function"] = memberAccess->memberName();
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
				if (memberAccess->memberName() == "call" || memberAccess->memberName() == "delegatecall")
				{
					Json result = Json::object();
					result["kind"] = "low_level_call";
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
						emptyData["kind"] = "u256";
						emptyData["value"] = "0";
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
						emptyData["kind"] = "u256";
						emptyData["value"] = "0";
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
			result["function"] = memberAccess->memberName();
		else
			result["function"] = "unknown_call";
		result["args"] = Json::array();
		for (auto const& arg: call->arguments())
		{
			try { result["args"].emplace_back(exportExpr(*arg)); }
			catch (...) {
				Json zero = Json::object();
				zero["kind"] = "u256";
				zero["value"] = "0";
				result["args"].emplace_back(zero);
			}
		}
		return result;
	}

	// Fallback for FunctionCallOptions (e.g., addr.call{value: 1}): export as its inner expression
	if (auto const* options = dynamic_cast<FunctionCallOptions const*>(&_expr))
		return exportExpr(options->expression());

	// Ultimate fallback — emit a zero placeholder instead of throwing.
	// This allows partial export of functions that use an unsupported expression
	// pattern in a subexpression, rather than failing the entire function.
	{
		Json result = Json::object();
		result["kind"] = "u256";
		result["value"] = "0";
		result["_unsupported"] = true;
		return result;
	}
}

Json exportRevertPayload(FunctionCall const& _call)
{
	Json result = Json::object();
	if (auto const* callee = dynamic_cast<Identifier const*>(&_call.expression()))
	{
		if (callee->name() == "require" || callee->name() == "revert")
		{
			if (_call.arguments().size() >= 2)
			{
				auto const* message = dynamic_cast<Literal const*>(_call.arguments().at(1).get());
				if (message && message->token() == Token::StringLiteral)
				{
					result["message"] = message->value();
					return result;
				}
				// Non-string revert message (e.g. custom error): export as generic expression
				try
				{
					result["payload"] = exportExpr(*_call.arguments().at(1));
				}
				catch (...)
				{
					result["message"] = "(non-string revert payload)";
				}
				return result;
			}
			return result;
		}
	}

	if (auto const* callee = dynamic_cast<Identifier const*>(&_call.expression()))
	{
		result["error"] = Json::object();
		result["error"]["name"] = callee->name();
		return result;
	}

	// Fallback: return empty payload rather than throwing
	return result;
}

Json exportStmt(Statement const& _stmt);

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
			break;
		case Token::AssignSub:
			value["kind"] = "u256_sub";
			break;
		case Token::AssignMul:
			value["kind"] = "u256_mul";
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
			// If mapping export fails, fall back to generic array_set_expr
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

	// Member access LHS: structVar.field = value
	if (auto const* memberAccess = dynamic_cast<MemberAccess const*>(&_lhs))
	{
		// Check if the base is a state variable (storage field set on a nested struct)
		if (auto const* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
		{
			auto const* decl = dynamic_cast<VariableDeclaration const*>(baseIdent->annotation().referencedDeclaration);
			if (decl && decl->isStateVariable())
			{
				// storage.structField.member = val → storage_set with field path
				Json result = Json::object();
				result["kind"] = "expr";
				Json callExpr = Json::object();
				callExpr["kind"] = "internal_call";
				callExpr["function"] = "field_set";
				callExpr["args"] = Json::array();
				callExpr["args"].emplace_back(exportExpr(memberAccess->expression()));
				Json fieldName = Json::object();
				fieldName["kind"] = "string";
				fieldName["value"] = memberAccess->memberName();
				callExpr["args"].emplace_back(fieldName);
				callExpr["args"].emplace_back(rhsJson);
				result["value"] = callExpr;
				return result;
			}
		}
		// Generic member access assignment: export as expr
		Json result = Json::object();
		result["kind"] = "expr";
		Json callExpr = Json::object();
		callExpr["kind"] = "internal_call";
		callExpr["function"] = "field_set";
		callExpr["args"] = Json::array();
		try
		{
			callExpr["args"].emplace_back(exportExpr(memberAccess->expression()));
		}
		catch (...)
		{
			Json zero = Json::object();
			zero["kind"] = "u256";
			zero["value"] = "0";
			callExpr["args"].emplace_back(zero);
		}
		Json fieldName = Json::object();
		fieldName["kind"] = "string";
		fieldName["value"] = memberAccess->memberName();
		callExpr["args"].emplace_back(fieldName);
		callExpr["args"].emplace_back(rhsJson);
		result["value"] = callExpr;
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
		for (auto const& component: tuple->components())
		{
			if (component)
			{
				try { callExpr["args"].emplace_back(exportExpr(*component)); }
				catch (...) {
					Json zero = Json::object();
					zero["kind"] = "u256";
					zero["value"] = "0";
					callExpr["args"].emplace_back(zero);
				}
			}
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
		try { callExpr["args"].emplace_back(exportExpr(_lhs)); }
		catch (...) {
			Json zero = Json::object();
			zero["kind"] = "u256";
			zero["value"] = "0";
			callExpr["args"].emplace_back(zero);
		}
		callExpr["args"].emplace_back(rhsJson);
		result["value"] = callExpr;
		return result;
	}
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
		Json result = Json::object();
		result["kind"] = "block";
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

				// Internal function call as statement
				auto const* funcDef = dynamic_cast<FunctionDefinition const*>(callee->annotation().referencedDeclaration);
				if (funcDef)
				{
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = callee->name();
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

				// External contract call as statement: contract.method(args)
				Type const* baseType = memberAccess->expression().annotation().type;
				if (baseType && baseType->category() == Type::Category::Contract)
				{
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "external_call";
					callExpr["target"] = exportExpr(memberAccess->expression());
					callExpr["method"] = memberAccess->memberName();
					callExpr["args"] = Json::array();
					for (auto const& arg: call->arguments())
						callExpr["args"].emplace_back(exportExpr(*arg));
					result["value"] = callExpr;
					return result;
				}

				// Library-qualified or type-qualified function call as statement: L.f(args)
				if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
				{
					(void)funcDef;
					Json result = Json::object();
					result["kind"] = "expr";
					Json callExpr = Json::object();
					callExpr["kind"] = "internal_call";
					callExpr["function"] = memberAccess->memberName();
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
					if (memberAccess->memberName() == "call" || memberAccess->memberName() == "delegatecall")
					{
						Json result = Json::object();
						result["kind"] = "expr";
						Json callExpr = Json::object();
						callExpr["kind"] = "low_level_call";
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
							emptyData["kind"] = "u256";
							emptyData["value"] = "0";
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
							emptyData["kind"] = "u256";
							emptyData["value"] = "0";
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

Json exportBody(FunctionDefinition const& _function)
{
	Json body = exportStmt(_function.body());
	if (!body.is_object() || body.value("kind", ""s) != "block")
		return body;

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
				letStmt["type"] = exportTypeName(retParam->typeName());
				letStmt["value"] = defaultValueForTypeName(retParam->typeName());
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
	return body;
}

Json exportFunction(FunctionDefinition const& _function, bool _isInternal = false)
{
	if (!_function.isOrdinary() || !_function.isImplemented())
		throw UnsupportedSolCore("Only ordinary implemented functions are supported.");
	if (!_isInternal && !(_function.visibility() == Visibility::Public || _function.visibility() == Visibility::External))
		throw UnsupportedSolCore("Only public/external functions are supported.");
	// Note: we no longer reject functions with modifiers or multiple return values.
	// Modifiers are ignored (only the function body is exported).
	// Multiple return values are exported as a tuple return type.

	Json result = Json::object();
	result["name"] = _function.name();
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
		entry["kind"] = "stateVariable";
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

namespace v0_8
{

solcore::ExportArtifacts exportContract(CompilerStack const& _compilerStack, std::string const& _contractName)
{
	ContractDefinition const& contract = _compilerStack.contractDefinition(_contractName);

	Json solcore = Json::object();
	solcore["solcoreVersion"] = "0.1.0";
	solcore["solidityVersion"] = VersionString;
	solcore["featureFlags"] = featureFlags();
	Json metadata = exporterMetadata(_contractName);
	for (auto const& [key, value]: metadata.items())
		solcore[key] = value;
	solcore["crate_name"] = contract.name();

	Json typeDecls = Json::array();
	Json storageFields = Json::array();
	for (VariableDeclaration const* stateVar: contract.stateVariables())
	{
		// Skip constant variables — they are inlined at usage sites
		if (stateVar->isConstant())
			continue;
		// Skip immutable variables whose value can be successfully inlined.
		// If inlining the value would fail (e.g., function pointer, unsupported expression),
		// the identifier handler falls back to storage_get, so we must include the field.
		if (stateVar->immutable() && stateVar->value())
		{
			try
			{
				(void)exportExpr(*stateVar->value());
				continue; // Inlining works — skip from Storage
			}
			catch (...)
			{
				// Inlining failed — include in Storage so storage_get can find it
			}
		}
		try
		{
			storageFields.emplace_back(exportField(*stateVar));
		}
		catch (UnsupportedSolCore const&)
		{
			// Skip state variables with unsupported types
		}
		catch (std::exception const&)
		{
			// Skip state variables that cause unexpected errors during export
		}
	}
	typeDecls.emplace_back(runtimeTypeDecl("Storage", std::move(storageFields)));

	Json callEnvFields = Json::array();
	callEnvFields.emplace_back(Json{{"name", "msgSender"}, {"type", "address"}});
	callEnvFields.emplace_back(Json{{"name", "msgValue"}, {"type", "u256"}});
	callEnvFields.emplace_back(Json{{"name", "blockTimestamp"}, {"type", "u256"}});
	callEnvFields.emplace_back(Json{{"name", "blockNumber"}, {"type", "u256"}});
	callEnvFields.emplace_back(Json{{"name", "thisAddress"}, {"type", "address"}});
	typeDecls.emplace_back(runtimeTypeDecl("CallEnv", std::move(callEnvFields)));
	Json worldStateFields = Json::array();
	worldStateFields.emplace_back(Json{{"name", "contractBalance"}, {"type", "u256"}});
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

	solcore["type_decls"] = std::move(typeDecls);

	Json state = Json::object();
	state["name"] = "ExecState";
	Json stateFields = Json::array();
	stateFields.emplace_back(Json{{"name", "world"}, {"type", Json{{"kind", "named"}, {"name", "WorldState"}}}});
	stateFields.emplace_back(Json{{"name", "storage"}, {"type", Json{{"kind", "named"}, {"name", "Storage"}}}});
	stateFields.emplace_back(Json{{"name", "memory"}, {"type", Json{{"kind", "named"}, {"name", "Memory"}}}});
	stateFields.emplace_back(Json{{"name", "returndata"}, {"type", Json{{"kind", "named"}, {"name", "ByteArray"}}}});
	stateFields.emplace_back(Json{{"name", "logs"}, {"type", Json{{"kind", "named"}, {"name", "Logs"}}}});
	stateFields.emplace_back(Json{{"name", "env"}, {"type", Json{{"kind", "named"}, {"name", "CallEnv"}}}});
	state["fields"] = std::move(stateFields);
	solcore["state"] = std::move(state);

	Json functions = Json::array();
	Json internalFunctions = Json::array();
	for (FunctionDefinition const* function: contract.definedFunctions())
	{
		if (!function->isOrdinary() || !function->isImplemented())
			continue;
		if (function->visibility() == Visibility::Public || function->visibility() == Visibility::External)
		{
			try
			{
				functions.emplace_back(exportFunction(*function));
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
				internalFunctions.emplace_back(exportFunction(*function, /*_isInternal=*/true));
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
				f["return"] = Json("unit");
				f["body"] = exportBody(*recv);
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
				f["return"] = Json("unit");
				f["body"] = exportBody(*fb);
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

	solcore["functions"] = std::move(functions);
	if (!internalFunctions.empty())
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

	// Export enum declarations (including inherited ones)
	Json enumDecls = Json::array();
	for (auto const& node : contract.sourceUnit().nodes())
	{
		if (auto const* enumDef = dynamic_cast<EnumDefinition const*>(node.get()))
		{
			Json decl = Json::object();
			decl["name"] = enumDef->name();
			decl["variants"] = Json::array();
			for (auto const& member : enumDef->members())
				decl["variants"].emplace_back(member->name());
			enumDecls.emplace_back(decl);
		}
	}
	// Also check enums defined inside contracts and interfaces in the same source unit
	for (auto const& node : contract.sourceUnit().nodes())
	{
		if (auto const* contractNode = dynamic_cast<ContractDefinition const*>(node.get()))
		{
			for (auto const& subNode : contractNode->subNodes())
			{
				if (auto const* enumDef = dynamic_cast<EnumDefinition const*>(subNode.get()))
				{
					Json decl = Json::object();
					decl["name"] = enumDef->name();
					decl["variants"] = Json::array();
					for (auto const& member : enumDef->members())
						decl["variants"].emplace_back(member->name());
					// Avoid duplicates
					bool exists = false;
					for (auto const& existing : enumDecls)
						if (existing.value("name", "") == enumDef->name())
							exists = true;
					if (!exists)
						enumDecls.emplace_back(decl);
				}
			}
		}
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
