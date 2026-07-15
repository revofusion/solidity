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
/**
 * @date 2017
 * Unit tests for interface/StandardCompiler.h.
 */

#include <string>
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <liblangutil/EVMVersion.h>
#include <libsolidity/interface/OptimiserSettings.h>
#include <libsolidity/interface/StandardCompiler.h>
#include <libsolidity/interface/Version.h>
#include <libsolutil/JSON.h>
#include <libsolutil/CommonData.h>
#include <test/Metadata.h>
#include <test/Common.h>

#include <algorithm>
#include <set>
#include <utility>

using namespace solidity::evmasm;
using namespace solidity::langutil;
using namespace std::string_literals;

namespace solidity::frontend::test
{

namespace
{

Error::Severity str2Severity(std::string const& _cat)
{
	std::map<std::string, Error::Severity> cats{
		{"info", Error::Severity::Info},
		{"Info", Error::Severity::Info},
		{"warning", Error::Severity::Warning},
		{"Warning", Error::Severity::Warning},
		{"error", Error::Severity::Error},
		{"Error", Error::Severity::Error}
	};
	return cats.at(_cat);
}

/// Helper to match a specific error type and message
bool containsError(Json const& _compilerResult, std::string const& _type, std::string const& _message)
{
	if (!_compilerResult.contains("errors"))
		return false;

	for (auto const& error: _compilerResult["errors"])
	{
		BOOST_REQUIRE(error.is_object());
		BOOST_REQUIRE(error["type"].is_string());
		BOOST_REQUIRE(error["message"].is_string());
		if ((error["type"].get<std::string>() == _type) && (error["message"].get<std::string>() == _message))
			return true;
	}

	return false;
}

bool containsAtMostWarnings(Json const& _compilerResult)
{
	if (!_compilerResult.contains("errors"))
		return true;

	for (auto const& error: _compilerResult["errors"])
	{
		BOOST_REQUIRE(error.is_object());
		BOOST_REQUIRE(error["severity"].is_string());
		if (Error::isError(str2Severity(error["severity"].get<std::string>())))
			return false;
	}

	return true;
}

Json getContractResult(Json const& _compilerResult, std::string const& _file, std::string const& _name)
{
	if (!_compilerResult.contains("contracts") ||
		!_compilerResult["contracts"].is_object() ||
		!_compilerResult["contracts"][_file].is_object() ||
		!_compilerResult["contracts"][_file][_name].is_object()
	)
		return Json();
	return _compilerResult["contracts"][_file][_name];
}

void checkLinkReferencesSchema(Json const& _contractResult)
{
	BOOST_TEST_REQUIRE(_contractResult.is_object());
	BOOST_TEST_REQUIRE(_contractResult["evm"]["bytecode"].is_object());

	Json const& linkReferenceResult = _contractResult["evm"]["bytecode"]["linkReferences"];
	BOOST_TEST_REQUIRE(linkReferenceResult.is_object());

	for (auto const& [fileName, references]: linkReferenceResult.items())
	{
		BOOST_TEST_REQUIRE(references.is_object());
		for (auto const& [libraryName, libraryValue]: references.items())
		{
			BOOST_TEST_REQUIRE(libraryValue.is_array());
			BOOST_TEST_REQUIRE(!libraryValue.empty());
			for (size_t i = 0; i < static_cast<size_t>(linkReferenceResult.size()); ++i)
			{
				BOOST_TEST_REQUIRE(libraryValue[i].is_object());
				BOOST_TEST_REQUIRE(libraryValue[i].size() == 2);
				BOOST_TEST_REQUIRE(libraryValue[i]["length"].is_number_unsigned());
				BOOST_TEST_REQUIRE(libraryValue[i]["start"].is_number_unsigned());
			}
		}
	}
}

void expectLinkReferences(Json const& _contractResult, std::map<std::string, std::set<std::string>> const& _expectedLinkReferences)
{
	checkLinkReferencesSchema(_contractResult);

	Json const& linkReferenceResult = _contractResult["evm"]["bytecode"]["linkReferences"];
	BOOST_TEST(linkReferenceResult.size() == _expectedLinkReferences.size());

	for (auto const& [fileName, libraries]: _expectedLinkReferences)
	{
		BOOST_TEST(linkReferenceResult.contains(fileName));
		BOOST_TEST(linkReferenceResult[fileName].size() == libraries.size());
		for (std::string const& libraryName: libraries)
			BOOST_TEST(linkReferenceResult[fileName].contains(libraryName));
	}
}

Json compile(std::string _input)
{
	StandardCompiler compiler;
	std::string output = compiler.compile(std::move(_input));
	Json ret;
	BOOST_REQUIRE(util::jsonParseStrict(output, ret));
	return ret;
}

Json createLanguageAndSourcesSection(std::string const& _language, std::map<std::string, Json> const& _sources, bool _contentNode = true)
{
	Json result = Json::object();
	result["language"] = _language;
	result["sources"] = Json::object();
	for (auto const& source: _sources)
	{
		result["sources"][source.first] = Json::object();
		if (_contentNode)
			result["sources"][source.first]["content"] = source.second;
		else
			result["sources"][source.first] = source.second;
	}
	return result;
}

class Code
{
public:
	virtual ~Code() = default;
	explicit Code(std::map<std::string, Json> _code = {}) : m_code(std::move(_code)) {}
	[[nodiscard]] virtual Json json() const = 0;
protected:
	std::map<std::string, Json> m_code;
};

class SolidityCode: public Code
{
public:
	explicit SolidityCode(std::map<std::string, Json> _code = {
		{"fileA", "pragma solidity >=0.0; contract C { function f() public pure {} }"}
	}) : Code(std::move(_code)) {}
	[[nodiscard]] Json json() const override
	{
		return createLanguageAndSourcesSection("Solidity", m_code);
	}
};

class YulCode: public Code
{
public:
	explicit YulCode(std::map<std::string, Json> _code = {
		{"fileA", "{}"}
	}) : Code(std::move(_code)) {}
	[[nodiscard]] Json json() const override
	{
		return createLanguageAndSourcesSection("Yul", m_code);
	}
};

class EvmAssemblyCode: public Code
{
public:
	explicit EvmAssemblyCode(std::map<std::string, Json> _code = {
		{"fileA", Json::parse(R"(
			{
				"assemblyJson": {
					".code": [
						{
							"begin": 36,
							"end": 51,
							"name": "PUSH",
							"source": 0,
							"value": "0"
						}
					],
					"sourceList": [
						"<stdin>"
					]
				}
			}
			)")}
	}) : Code(std::move(_code)) {}
	[[nodiscard]] Json json() const override
	{
		return createLanguageAndSourcesSection("EVMAssembly", m_code, false);
	}
};

class SolidityAstCode: public Code
{
public:
	explicit SolidityAstCode(std::map<std::string, Json> _code = {
		{"fileA", Json::parse(R"(
		{
			"ast": {
				"absolutePath": "empty_contract.sol",
				"exportedSymbols": {
					"test": [
						1
					]
				},
				"id": 2,
				"nodeType": "SourceUnit",
				"nodes": [
				{
					"abstract": false,
					"baseContracts": [],
					"canonicalName": "test",
					"contractDependencies": [],
					"contractKind": "contract",
					"fullyImplemented": true,
					"id": 1,
					"linearizedBaseContracts": [
						1
					],
					"name": "test",
					"nameLocation": "9:4:0",
					"nodeType": "ContractDefinition",
					"nodes": [],
					"scope": 2,
					"src": "0:17:0",
					"usedErrors": []
				}
				],
			"src": "0:124:0"
			},
			"id": 0
		}
		)")}
	}) : Code(std::move(_code)) {}
	[[nodiscard]] Json json() const override
	{
		return createLanguageAndSourcesSection("SolidityAST", m_code);
	}
};

Json generateStandardJson(bool _viaIr, Json const& _debugInfoSelection, Json const& _outputSelection, Code const& _code = SolidityCode(), bool _advancedOutputSelection = false)
{
	Json result = _code.json();
	result["settings"] = Json::object();
	result["settings"]["viaIR"] = _viaIr;
	if (!_debugInfoSelection.empty())
		result["settings"]["debug"]["debugInfo"] = _debugInfoSelection;
	if (_advancedOutputSelection)
		result["settings"]["outputSelection"] = _outputSelection;
	else
		result["settings"]["outputSelection"]["*"]["*"] = _outputSelection;
	return result;
}

Json generateExperimentalStandardJson(bool _viaIR, Json const& _debugInfoSelection, Json const& _outputSelection, Code const& _code = SolidityCode(), bool _advancedOutputSelection = false)
{
	Json result = generateStandardJson(_viaIR, _debugInfoSelection, _outputSelection, _code, _advancedOutputSelection);
	result["settings"]["experimental"] = true;
	return result;
}

} // end anonymous namespace

BOOST_AUTO_TEST_SUITE(StandardCompiler)

BOOST_AUTO_TEST_CASE(assume_object_input)
{
	Json result;

	/// Use the native JSON interface of StandardCompiler to trigger these
	frontend::StandardCompiler compiler;
	result = compiler.compile(Json());
	BOOST_CHECK(containsError(result, "JSONError", "Input is not a JSON object."));
	result = compiler.compile(Json("INVALID"));
	BOOST_CHECK(containsError(result, "JSONError", "Input is not a JSON object."));

	/// Use the string interface of StandardCompiler to trigger these
	result = compile("");
	BOOST_CHECK(containsError(result, "JSONError", "parse error at line 1, column 1: attempting to parse an empty input; check that your input string or stream contains the expected JSON"));
	result = compile("invalid");
	BOOST_CHECK(containsError(result, "JSONError", "parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'i'"));
	result = compile("\"invalid\"");
	BOOST_CHECK(containsError(result, "JSONError", "Input is not a JSON object."));
	result = compile("{}");
	BOOST_CHECK(containsError(result, "JSONError", "No input sources specified."));
	BOOST_CHECK(!containsAtMostWarnings(result));
}

BOOST_AUTO_TEST_CASE(invalid_language)
{
	char const* input = R"(
	{
		"language": "INVALID",
		"sources": { "name": { "content": "abc" } }
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "Only \"Solidity\", \"Yul\", \"SolidityAST\" or \"EVMAssembly\" is supported as a language."));
}

BOOST_AUTO_TEST_CASE(valid_language)
{
	char const* input = R"(
	{
		"language": "Solidity"
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(!containsError(result, "JSONError", "Only \"Solidity\" or \"Yul\" is supported as a language."));
}

BOOST_AUTO_TEST_CASE(no_sources)
{
	char const* input = R"(
	{
		"language": "Solidity"
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "No input sources specified."));
}

BOOST_AUTO_TEST_CASE(no_sources_empty_object)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "No input sources specified."));
}

BOOST_AUTO_TEST_CASE(no_sources_empty_array)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": []
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "\"sources\" is not a JSON object."));
}

BOOST_AUTO_TEST_CASE(sources_is_array)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": ["aa", "bb"]
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "\"sources\" is not a JSON object."));
}

BOOST_AUTO_TEST_CASE(unexpected_trailing_test)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"A": {
				"content": "contract A { function f() {} }"
			}
		}
	}
	}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "parse error at line 10, column 2: syntax error while parsing value - unexpected '}'; expected end of input"));
}

BOOST_AUTO_TEST_CASE(smoke_test)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
}

BOOST_AUTO_TEST_CASE(optimizer_enabled_not_boolean)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"optimizer": {
				"enabled": "wrong"
			}
		},
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "The \"enabled\" setting must be a Boolean."));
}

BOOST_AUTO_TEST_CASE(optimizer_runs_not_a_number)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"optimizer": {
				"enabled": true,
				"runs": "not a number"
			}
		},
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "The \"runs\" setting must be an unsigned number."));
}

BOOST_AUTO_TEST_CASE(optimizer_runs_not_an_unsigned_number)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"optimizer": {
				"enabled": true,
				"runs": -1
			}
		},
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "The \"runs\" setting must be an unsigned number."));
}

BOOST_AUTO_TEST_CASE(basic_compilation)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		},
		"settings": {
			"outputSelection": {
				"fileA": {
					"A": [ "abi", "devdoc", "userdoc", "evm.bytecode", "evm.assembly", "evm.gasEstimates", "evm.legacyAssembly", "metadata" ],
					"": [ "ast" ]
				}
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["abi"].is_array());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["abi"]), "[]");
	BOOST_CHECK(contract["devdoc"].is_object());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["devdoc"]), R"({"kind":"dev","methods":{},"version":1})");
	BOOST_CHECK(contract["userdoc"].is_object());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["userdoc"]), R"({"kind":"user","methods":{},"version":1})");
	BOOST_CHECK(contract["evm"].is_object());
	/// @TODO check evm.methodIdentifiers, legacyAssembly, bytecode, deployedBytecode
	BOOST_CHECK(contract["evm"]["bytecode"].is_object());
	BOOST_CHECK(contract["evm"]["bytecode"]["object"].is_string());
	BOOST_CHECK_EQUAL(
		solidity::test::bytecodeSansMetadata(contract["evm"]["bytecode"]["object"].get<std::string>()),
		std::string("6080604052348015600e575f5ffd5b5060") +
		(VersionIsRelease ? "3e" : util::toHex(bytes{uint8_t(60 + VersionStringStrict.size())})) +
		"80601a5f395ff3fe60806040525f5ffdfe"
	);
	BOOST_CHECK(contract["evm"]["assembly"].is_string());
	BOOST_CHECK(contract["evm"]["assembly"].get<std::string>().find(
		"    /* \"fileA\":0:14  contract A { } */\n  mstore(0x40, 0x80)\n  "
		"callvalue\n  dup1\n  "
		"iszero\n  tag_1\n  jumpi\n  "
		"revert(0x00, 0x00)\n"
		"tag_1:\n  pop\n  dataSize(sub_0)\n  dup1\n  "
		"dataOffset(sub_0)\n  0x00\n  codecopy\n  0x00\n  return\nstop\n\nsub_0: assembly {\n        "
		"/* \"fileA\":0:14  contract A { } */\n      mstore(0x40, 0x80)\n      "
		"revert(0x00, 0x00)\n\n    auxdata: 0xa26469706673582212"
	) == 0);
	BOOST_CHECK(contract["evm"]["gasEstimates"].is_object());
	BOOST_CHECK_EQUAL(contract["evm"]["gasEstimates"].size(), 1);
	BOOST_CHECK(contract["evm"]["gasEstimates"]["creation"].is_object());
	BOOST_CHECK_EQUAL(contract["evm"]["gasEstimates"]["creation"].size(), 3);
	BOOST_CHECK(contract["evm"]["gasEstimates"]["creation"]["codeDepositCost"].is_string());
	BOOST_CHECK(contract["evm"]["gasEstimates"]["creation"]["executionCost"].is_string());
	BOOST_CHECK(contract["evm"]["gasEstimates"]["creation"]["totalCost"].is_string());
	BOOST_CHECK_EQUAL(
		u256(contract["evm"]["gasEstimates"]["creation"]["codeDepositCost"].get<std::string>()) +
		u256(contract["evm"]["gasEstimates"]["creation"]["executionCost"].get<std::string>()),
		u256(contract["evm"]["gasEstimates"]["creation"]["totalCost"].get<std::string>())
	);
	// Lets take the top level `.code` section (the "deployer code"), that should expose most of the features of
	// the assembly JSON. What we want to check here is Operation, Push, PushTag, PushSub, PushSubSize and Tag.
	BOOST_CHECK(contract["evm"]["legacyAssembly"].is_object());
	BOOST_CHECK(contract["evm"]["legacyAssembly"][".code"].is_array());
	BOOST_CHECK_EQUAL(
		util::jsonCompactPrint(contract["evm"]["legacyAssembly"][".code"]),
		"[{\"begin\":0,\"end\":14,\"name\":\"PUSH\",\"source\":0,\"value\":\"80\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"PUSH\",\"source\":0,\"value\":\"40\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"MSTORE\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"CALLVALUE\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"DUP1\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"ISZERO\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"PUSH [tag]\",\"source\":0,\"value\":\"1\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"JUMPI\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"PUSH\",\"source\":0,\"value\":\"0\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"PUSH\",\"source\":0,\"value\":\"0\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"REVERT\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"tag\",\"source\":0,\"value\":\"1\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"JUMPDEST\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"POP\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"PUSH #[$]\",\"source\":0,\"value\":\"0000000000000000000000000000000000000000000000000000000000000000\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"DUP1\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"PUSH [$]\",\"source\":0,\"value\":\"0000000000000000000000000000000000000000000000000000000000000000\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"PUSH\",\"source\":0,\"value\":\"0\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"CODECOPY\",\"source\":0},"
		"{\"begin\":0,\"end\":14,\"name\":\"PUSH\",\"source\":0,\"value\":\"0\"},"
		"{\"begin\":0,\"end\":14,\"name\":\"RETURN\",\"source\":0}]"
	);
	BOOST_CHECK(contract["metadata"].is_string());
	BOOST_CHECK(solidity::test::isValidMetadata(contract["metadata"].get<std::string>()));
	BOOST_CHECK(result["sources"].is_object());
	BOOST_CHECK(result["sources"]["fileA"].is_object());
	BOOST_CHECK(result["sources"]["fileA"]["ast"].is_object());
	BOOST_CHECK_EQUAL(
		util::jsonCompactPrint(result["sources"]["fileA"]["ast"]),
		"{\"absolutePath\":\"fileA\",\"exportedSymbols\":{\"A\":[1]},\"id\":2,\"nodeType\":\"SourceUnit\",\"nodes\":[{\"abstract\":false,"
		"\"baseContracts\":[],\"canonicalName\":\"A\",\"contractDependencies\":[],"
		"\"contractKind\":\"contract\",\"fullyImplemented\":true,\"id\":1,"
		"\"linearizedBaseContracts\":[1],\"name\":\"A\",\"nameLocation\":\"9:1:0\",\"nodeType\":\"ContractDefinition\",\"nodes\":[],\"scope\":2,"
		"\"src\":\"0:14:0\",\"usedErrors\":[],\"usedEvents\":[]}],\"src\":\"0:14:0\"}"
	);
}

BOOST_AUTO_TEST_CASE(compilation_error)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": {
					"A": [
						"abi"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A { function }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(result.contains("errors"));
	BOOST_CHECK(result["errors"].size() >= 1);
	for (auto const& error: result["errors"])
	{
		BOOST_REQUIRE(error.is_object());
		BOOST_REQUIRE(error["message"].is_string());
		if (error["message"].get<std::string>().find("pre-release compiler") == std::string::npos)
		{
			BOOST_CHECK_EQUAL(
				util::jsonCompactPrint(error),
				"{\"component\":\"general\",\"errorCode\":\"2314\",\"formattedMessage\":\"ParserError: Expected identifier but got '}'\\n"
				" --> fileA:1:23:\\n  |\\n1 | contract A { function }\\n  |                       ^\\n\\n\",\"message\":\"Expected identifier but got '}'\","
				"\"severity\":\"error\",\"sourceLocation\":{\"end\":23,\"file\":\"fileA\",\"start\":22},\"type\":\"ParserError\"}"
			);
		}
	}
}

BOOST_AUTO_TEST_CASE(output_selection_explicit)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": {
					"A": [
						"abi"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["abi"].is_array());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["abi"]), "[]");
}

BOOST_AUTO_TEST_CASE(output_selection_all_contracts)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": {
					"*": [
						"abi"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["abi"].is_array());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["abi"]), "[]");
}

BOOST_AUTO_TEST_CASE(output_selection_all_files_single_contract)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"*": {
					"A": [
						"abi"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["abi"].is_array());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["abi"]), "[]");
}

BOOST_AUTO_TEST_CASE(output_selection_all_files_all_contracts)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"*": {
					"*": [
						"abi"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["abi"].is_array());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["abi"]), "[]");
}

BOOST_AUTO_TEST_CASE(output_selection_dependent_contract)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"*": {
					"A": [
						"abi"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "contract B { } contract A { function f() public { new B(); } }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["abi"].is_array());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["abi"]), "[{\"inputs\":[],\"name\":\"f\",\"outputs\":[],\"stateMutability\":\"nonpayable\",\"type\":\"function\"}]");
}

BOOST_AUTO_TEST_CASE(output_selection_dependent_contract_with_import)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"*": {
					"A": [
						"abi"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "import \"fileB\"; contract A { function f() public { new B(); } }"
			},
			"fileB": {
				"content": "contract B { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["abi"].is_array());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["abi"]), "[{\"inputs\":[],\"name\":\"f\",\"outputs\":[],\"stateMutability\":\"nonpayable\",\"type\":\"function\"}]");
}

BOOST_AUTO_TEST_CASE(filename_with_colon)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"https://github.com/argotorg/solidity/blob/develop/test/compilationTests/gnosis/Tokens/StandardToken.sol": {
					"A": [
						"abi"
					]
				}
			}
		},
		"sources": {
			"https://github.com/argotorg/solidity/blob/develop/test/compilationTests/gnosis/Tokens/StandardToken.sol": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "https://github.com/argotorg/solidity/blob/develop/test/compilationTests/gnosis/Tokens/StandardToken.sol", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["abi"].is_array());
	BOOST_CHECK_EQUAL(util::jsonCompactPrint(contract["abi"]), "[]");
}

BOOST_AUTO_TEST_CASE(library_filename_with_colon)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": {
					"A": [
						"evm.bytecode"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "import \"git:library.sol\"; contract A { function f() public returns (uint) { return L.g(); } }"
			},
			"git:library.sol": {
				"content": "library L { function g() public returns (uint) { return 1; } }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	expectLinkReferences(contract, {{"git:library.sol", {"L"}}});
}

BOOST_AUTO_TEST_CASE(libraries_invalid_top_level)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"libraries": "42"
		},
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "\"libraries\" is not a JSON object."));
}

BOOST_AUTO_TEST_CASE(libraries_invalid_entry)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"libraries": {
				"L": "42"
			}
		},
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "Library entry is not a JSON object."));
}

BOOST_AUTO_TEST_CASE(libraries_invalid_hex)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"libraries": {
				"library.sol": {
					"L": "0x4200000000000000000000000000000000000xx1"
				}
			}
		},
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "Invalid library address (\"0x4200000000000000000000000000000000000xx1\") supplied."));
}

BOOST_AUTO_TEST_CASE(libraries_invalid_length)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"libraries": {
				"library.sol": {
					"L1": "0x42",
					"L2": "0x4200000000000000000000000000000000000001ff"
				}
			}
		},
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "Library address is of invalid length."));
}

BOOST_AUTO_TEST_CASE(libraries_missing_hex_prefix)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"libraries": {
				"library.sol": {
					"L": "4200000000000000000000000000000000000001"
				}
			}
		},
		"sources": {
			"empty": {
				"content": ""
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "Library address is not prefixed with \"0x\"."));
}

BOOST_AUTO_TEST_CASE(library_linking)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"libraries": {
				"library.sol": {
					"L": "0x4200000000000000000000000000000000000001"
				}
			},
			"outputSelection": {
				"fileA": {
					"A": [
						"evm.bytecode"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "import \"library.sol\"; import \"library2.sol\"; contract A { function f() public returns (uint) { L2.g(); return L.g(); } }"
			},
			"library.sol": {
				"content": "library L { function g() public returns (uint) { return 1; } }"
			},
			"library2.sol": {
				"content": "library L2 { function g() public { } }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_TEST(containsAtMostWarnings(result));
	Json contractResult = getContractResult(result, "fileA", "A");
	expectLinkReferences(contractResult, {{"library2.sol", {"L2"}}});
}

BOOST_AUTO_TEST_CASE(linking_yul)
{
	char const* input = R"(
	{
		"language": "Yul",
		"settings": {
			"libraries": {
				"fileB": {
					"L": "0x4200000000000000000000000000000000000001"
				}
			},
			"outputSelection": {
				"fileA": {
					"*": [
						"evm.bytecode.linkReferences"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "object \"a\" { code { let addr := linkersymbol(\"fileB:L\") sstore(0, addr) } }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_TEST(containsAtMostWarnings(result));
	Json contractResult = getContractResult(result, "fileA", "a");
	expectLinkReferences(contractResult, {});
}

BOOST_AUTO_TEST_CASE(linking_yul_empty_link_reference)
{
	char const* input = R"(
	{
		"language": "Yul",
		"settings": {
			"libraries": {
				"": {
					"": "0x4200000000000000000000000000000000000001"
				}
			},
			"outputSelection": {
				"fileA": {
					"*": [
						"evm.bytecode.linkReferences"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "object \"a\" { code { let addr := linkersymbol(\"\") sstore(0, addr) } }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_TEST(containsAtMostWarnings(result));
	Json contractResult = getContractResult(result, "fileA", "a");
	expectLinkReferences(contractResult, {{"", {""}}});
}

BOOST_AUTO_TEST_CASE(linking_yul_no_filename_in_link_reference)
{
	char const* input = R"(
	{
		"language": "Yul",
		"settings": {
			"libraries": {
				"": {
					"L": "0x4200000000000000000000000000000000000001"
				}
			},
			"outputSelection": {
				"fileA": {
					"*": [
						"evm.bytecode.linkReferences"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "object \"a\" { code { let addr := linkersymbol(\"L\") sstore(0, addr) } }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_TEST(containsAtMostWarnings(result));
	Json contractResult = getContractResult(result, "fileA", "a");
	expectLinkReferences(contractResult, {{"", {"L"}}});
}

BOOST_AUTO_TEST_CASE(linking_yul_same_library_name_different_files)
{
	char const* input = R"(
	{
		"language": "Yul",
		"settings": {
			"libraries": {
				"fileB": {
					"L": "0x4200000000000000000000000000000000000001"
				}
			},
			"outputSelection": {
				"fileA": {
					"*": [
						"evm.bytecode.linkReferences"
					]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "object \"a\" { code { let addr := linkersymbol(\"fileC:L\") sstore(0, addr) } }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_TEST(containsAtMostWarnings(result));
	Json contractResult = getContractResult(result, "fileA", "a");
	expectLinkReferences(contractResult, {{"fileC", {"L"}}});
}

BOOST_AUTO_TEST_CASE(evm_version)
{
	auto inputForVersion = [](std::string const& _version)
	{
		return R"(
			{
				"language": "Solidity",
				"sources": { "fileA": { "content": "contract A { }" } },
				"settings": {
					)" + _version + R"(
					"outputSelection": {
						"fileA": {
							"A": [ "metadata" ]
						}
					}
				}
			}
		)";
	};
	Json result;
	for (auto const& version: EVMVersion::allVersions())
	{
		result = compile(inputForVersion(fmt::format("\"evmVersion\": \"{}\",", version.name())));
		BOOST_CHECK(result["contracts"]["fileA"]["A"]["metadata"].get<std::string>().find(fmt::format("\"evmVersion\":\"{}\"", version.name())) != std::string::npos);
	}
	// test default
	result = compile(inputForVersion(""));
	BOOST_CHECK(result["contracts"]["fileA"]["A"]["metadata"].get<std::string>().find(fmt::format("\"evmVersion\":\"{}\"", EVMVersion::current().name())) != std::string::npos);
	// test invalid
	result = compile(inputForVersion("\"evmVersion\": \"invalid\","));
	BOOST_CHECK(result["errors"][0]["message"].get<std::string>() == "Invalid EVM version requested.");
}

BOOST_AUTO_TEST_CASE(optimizer_settings_default_disabled)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": { "A": [ "metadata" ] }
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["metadata"].is_string());
	Json metadata;
	BOOST_CHECK(util::jsonParseStrict(contract["metadata"].get<std::string>(), metadata));

	Json const& optimizer = metadata["settings"]["optimizer"];
	BOOST_CHECK(optimizer.contains("enabled"));
	BOOST_CHECK(optimizer["enabled"].get<bool>() == false);
	BOOST_CHECK(!optimizer.contains("details"));
	BOOST_CHECK(optimizer["runs"].get<unsigned>() == 200);
}

BOOST_AUTO_TEST_CASE(optimizer_settings_default_enabled)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": { "A": [ "metadata" ] }
			},
			"optimizer": { "enabled": true }
		},
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["metadata"].is_string());
	Json metadata;
	BOOST_CHECK(util::jsonParseStrict(contract["metadata"].get<std::string>(), metadata));

	Json const& optimizer = metadata["settings"]["optimizer"];
	BOOST_CHECK(optimizer.contains("enabled"));
	BOOST_CHECK(optimizer["enabled"].get<bool>() == true);
	BOOST_CHECK(!optimizer.contains("details"));
	BOOST_CHECK(optimizer["runs"].get<unsigned>() == 200);
}

BOOST_AUTO_TEST_CASE(optimizer_settings_details_exactly_as_default_disabled)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": { "A": [ "metadata" ] }
			},
			"optimizer": { "details": {
				"constantOptimizer" : false,
				"cse" : false,
				"deduplicate" : false,
				"jumpdestRemover" : true,
				"orderLiterals" : false,
				"peephole" : true
			} }
		},
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["metadata"].is_string());
	Json metadata;
	BOOST_CHECK(util::jsonParseStrict(contract["metadata"].get<std::string>(), metadata));

	Json const& optimizer = metadata["settings"]["optimizer"];
	BOOST_CHECK(optimizer.contains("enabled"));
	// enabled is switched to false instead!
	BOOST_CHECK(optimizer["enabled"].get<bool>() == false);
	BOOST_CHECK(!optimizer.contains("details"));
	BOOST_CHECK(optimizer["runs"].get<unsigned>() == 200);
}

BOOST_AUTO_TEST_CASE(optimizer_settings_details_different)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": { "A": [ "metadata" ] }
			},
			"optimizer": { "runs": 600, "details": {
				"constantOptimizer" : true,
				"cse" : false,
				"deduplicate" : true,
				"jumpdestRemover" : true,
				"orderLiterals" : false,
				"peephole" : true,
				"yul": true,
				"inliner": true
			} }
		},
		"sources": {
			"fileA": {
				"content": "contract A { }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["metadata"].is_string());
	Json metadata;
	BOOST_CHECK(util::jsonParseStrict(contract["metadata"].get<std::string>(), metadata));

	Json const& optimizer = metadata["settings"]["optimizer"];
	BOOST_CHECK(!optimizer.contains("enabled"));
	BOOST_CHECK(optimizer.contains("details"));
	BOOST_CHECK(optimizer["details"]["constantOptimizer"].get<bool>() == true);
	BOOST_CHECK(optimizer["details"]["cse"].get<bool>() == false);
	BOOST_CHECK(optimizer["details"]["deduplicate"].get<bool>() == true);
	BOOST_CHECK(optimizer["details"]["jumpdestRemover"].get<bool>() == true);
	BOOST_CHECK(optimizer["details"]["orderLiterals"].get<bool>() == false);
	BOOST_CHECK(optimizer["details"]["peephole"].get<bool>() == true);
	BOOST_CHECK(optimizer["details"]["yul"].get<bool>() == true);
	BOOST_CHECK(optimizer["details"]["yulDetails"].is_object());
//	BOOST_CHECK(
//		util::convertContainer<std::set<std::string>>(optimizer["details"]["yulDetails"].getMemberNames()) ==
//		(std::set<std::string>{"stackAllocation", "optimizerSteps"})
//	);
	BOOST_CHECK(optimizer["details"]["yulDetails"]["stackAllocation"].get<bool>() == true);
	BOOST_CHECK(
		optimizer["details"]["yulDetails"]["optimizerSteps"].get<std::string>() ==
		OptimiserSettings::DefaultYulOptimiserSteps + ":"s + OptimiserSettings::DefaultYulOptimiserCleanupSteps
 	);
	BOOST_CHECK_EQUAL(optimizer["details"].size(), 10);
	BOOST_CHECK(optimizer["runs"].get<unsigned>() == 600);
}

BOOST_AUTO_TEST_CASE(metadata_without_compilation)
{
	// NOTE: the contract code here should fail to compile due to "out of stack"
	// If the metadata is successfully returned, that means no compilation was attempted.
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"fileA": { "A": [ "metadata" ] }
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A {
  function x(uint a, uint b, uint c, uint d, uint e, uint f, uint g, uint h, uint i, uint j, uint k, uint l, uint m, uint n, uint o, uint p) pure public {}
  function y() pure public {
    uint a; uint b; uint c; uint d; uint e; uint f; uint g; uint h; uint i; uint j; uint k; uint l; uint m; uint n; uint o; uint p;
    x(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
  }
}"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["metadata"].is_string());
	BOOST_CHECK(solidity::test::isValidMetadata(contract["metadata"].get<std::string>()));
}


BOOST_AUTO_TEST_CASE(license_in_metadata)
{
	std::string const input = R"(
			{
				"language": "Solidity",
				"sources": {
					"fileA": { "content": "import \"fileB\"; contract A { } // SPDX-License-Identifier: GPL-3.0 \n" },
					"fileB": { "content": "import \"fileC\"; /* SPDX-License-Identifier: MIT */ contract B { }" },
					"fileC": { "content": "import \"fileD\"; /* SPDX-License-Identifier: MIT AND GPL-3.0 */ contract C { }" },
					"fileD": { "content": "// SPDX-License-Identifier: (GPL-3.0+ OR MIT) AND MIT \n import \"fileE\"; contract D { }" },
					"fileE": { "content": "import \"fileF\"; /// SPDX-License-Identifier: MIT   \n contract E { }" },
					"fileF": { "content": "/*\n * SPDX-License-Identifier: MIT\n */ contract F { }" }
				},
				"settings": {
					"outputSelection": {
						"fileA": {
							"*": [ "metadata" ]
						}
					}
				}
			}
		)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["metadata"].is_string());
	Json metadata;
	BOOST_REQUIRE(util::jsonParseStrict(contract["metadata"].get<std::string>(), metadata));
	BOOST_CHECK_EQUAL(metadata["sources"]["fileA"]["license"], "GPL-3.0");
	BOOST_CHECK_EQUAL(metadata["sources"]["fileB"]["license"], "MIT");
	BOOST_CHECK_EQUAL(metadata["sources"]["fileC"]["license"], "MIT AND GPL-3.0");
	BOOST_CHECK_EQUAL(metadata["sources"]["fileD"]["license"], "(GPL-3.0+ OR MIT) AND MIT");
	// This is actually part of the docstring, but still picked up
	// because the source location of the contract does not cover the docstring.
	BOOST_CHECK_EQUAL(metadata["sources"]["fileE"]["license"], "MIT");
	BOOST_CHECK_EQUAL(metadata["sources"]["fileF"]["license"], "MIT");
}

BOOST_AUTO_TEST_CASE(common_pattern)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"outputSelection": {
				"*": {
					"*": [ "evm.bytecode.object", "metadata" ]
				}
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A { function f() pure public {} }"
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_CHECK(contract.is_object());
	BOOST_CHECK(contract["metadata"].is_string());
	BOOST_CHECK(solidity::test::isValidMetadata(contract["metadata"].get<std::string>()));
	BOOST_CHECK(contract["evm"]["bytecode"].is_object());
	BOOST_CHECK(contract["evm"]["bytecode"]["object"].is_string());
}

BOOST_AUTO_TEST_CASE(use_stack_optimization)
{
	// NOTE: the contract code here should fail to compile due to "out of stack"
	// If we enable stack optimization, though, it will compile.
	char const* input = R"(
	{
		"language": "Solidity",
		"settings": {
			"optimizer": { "enabled": true, "details": { "yul": true } },
			"outputSelection": {
				"fileA": { "A": [ "evm.bytecode.object" ] }
			}
		},
		"sources": {
			"fileA": {
				"content": "contract A {
					function y() public {
						assembly {
							function fun() -> a3, b3, c3, d3, e3, f3, g3, h3, i3, j3, k3, l3, m3, n3, o3, p3
							{
								let a := 1
								let b := 1
								let z3 := 1
								sstore(a, b)
								sstore(add(a, 1), b)
								sstore(add(a, 2), b)
								sstore(add(a, 3), b)
								sstore(add(a, 4), b)
								sstore(add(a, 5), b)
								sstore(add(a, 6), b)
								sstore(add(a, 7), b)
								sstore(add(a, 8), b)
								sstore(add(a, 9), b)
								sstore(add(a, 10), b)
								sstore(add(a, 11), b)
								sstore(add(a, 12), b)
							}
							let a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1, o1, p1 := fun()
							let a2, b2, c2, d2, e2, f2, g2, h2, i2, j2, k2, l2, m2, n2, o2, p2 := fun()
							sstore(a1, a2)
						}
					}
				}"
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));

	solidity::frontend::StandardCompiler compiler;
	Json result = compiler.compile(parsedInput);

	BOOST_CHECK(containsAtMostWarnings(result));
	Json contract = getContractResult(result, "fileA", "A");
	BOOST_REQUIRE(contract.is_object());
	BOOST_REQUIRE(contract["evm"]["bytecode"]["object"].is_string());
	BOOST_CHECK(contract["evm"]["bytecode"]["object"].get<std::string>().length() > 20);

	// Now disable stack optimizations and UnusedFunctionParameterPruner (p)
	// results in "stack too deep"
	std::string optimiserSteps = OptimiserSettings::DefaultYulOptimiserSteps;
	optimiserSteps.erase(
		remove_if(optimiserSteps.begin(), optimiserSteps.end(), [](char ch) { return ch == 'p'; }),
		optimiserSteps.end()
	);
	parsedInput["settings"]["optimizer"]["details"]["yulDetails"]["stackAllocation"] = false;
	parsedInput["settings"]["optimizer"]["details"]["yulDetails"]["optimizerSteps"] = optimiserSteps;

	result = compiler.compile(parsedInput);
	BOOST_REQUIRE(result["errors"].is_array());
	BOOST_CHECK(result["errors"][0]["severity"] == "error");
	BOOST_REQUIRE(result["errors"][0]["message"].is_string());
	BOOST_CHECK(result["errors"][0]["message"].get<std::string>().find("When compiling inline assembly") != std::string::npos);
	BOOST_CHECK(result["errors"][0]["type"] == "CompilerError");
}

BOOST_AUTO_TEST_CASE(standard_output_selection_wildcard)
{
	char const* input = R"(
	{
		"language": "Solidity",
			"sources":
		{
			"A":
			{
				"content": "pragma solidity >=0.0; contract C { function f() public pure {} }"
			}
		},
		"settings":
		{
			"outputSelection":
			{
				"*": { "C": ["evm.bytecode"] }
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));

	solidity::frontend::StandardCompiler compiler;
	Json result = compiler.compile(parsedInput);

	BOOST_REQUIRE(result["contracts"].is_object());
	BOOST_REQUIRE(result["contracts"].size() == 1);
	BOOST_REQUIRE(result["contracts"]["A"].is_object());
	BOOST_REQUIRE(result["contracts"]["A"].size() == 1);
	BOOST_REQUIRE(result["contracts"]["A"]["C"].is_object());
	BOOST_REQUIRE(result["contracts"]["A"]["C"]["evm"].is_object());
	BOOST_REQUIRE(result["contracts"]["A"]["C"]["evm"]["bytecode"].is_object());
	BOOST_REQUIRE(result["sources"].is_object());
	BOOST_REQUIRE(result["sources"].size() == 1);
	BOOST_REQUIRE(result["sources"]["A"].is_object());

}

BOOST_AUTO_TEST_CASE(standard_output_selection_wildcard_colon_source)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources":
		{
			":A":
			{
				"content": "pragma solidity >=0.0; contract C { function f() public pure {} }"
			}
		},
		"settings":
		{
			"outputSelection":
			{
				"*": { "C": ["evm.bytecode"] }
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));

	solidity::frontend::StandardCompiler compiler;
	Json result = compiler.compile(parsedInput);

	BOOST_REQUIRE(result["contracts"].is_object());
	BOOST_REQUIRE(result["contracts"].size() == 1);
	BOOST_REQUIRE(result["contracts"][":A"].is_object());
	BOOST_REQUIRE(result["contracts"][":A"].size() == 1);
	BOOST_REQUIRE(result["contracts"][":A"]["C"].is_object());
	BOOST_REQUIRE(result["contracts"][":A"]["C"]["evm"].is_object());
	BOOST_REQUIRE(result["contracts"][":A"]["C"]["evm"]["bytecode"].is_object());
	BOOST_REQUIRE(result["sources"].is_object());
	BOOST_REQUIRE(result["sources"].size() == 1);
	BOOST_REQUIRE(result["sources"][":A"].is_object());
}

BOOST_AUTO_TEST_CASE(standard_output_selection_wildcard_empty_source)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources":
		{
			"":
			{
				"content": "pragma solidity >=0.0; contract C { function f() public pure {} }"
			}
		},
		"settings":
		{
			"outputSelection":
			{
				"*": { "C": ["evm.bytecode"] }
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));

	solidity::frontend::StandardCompiler compiler;
	Json result = compiler.compile(parsedInput);

	BOOST_REQUIRE(result["contracts"].is_object());
	BOOST_REQUIRE(result["contracts"].size() == 1);
	BOOST_REQUIRE(result["contracts"][""].is_object());
	BOOST_REQUIRE(result["contracts"][""].size() == 1);
	BOOST_REQUIRE(result["contracts"][""]["C"].is_object());
	BOOST_REQUIRE(result["contracts"][""]["C"]["evm"].is_object());
	BOOST_REQUIRE(result["contracts"][""]["C"]["evm"]["bytecode"].is_object());
	BOOST_REQUIRE(result["sources"].is_object());
	BOOST_REQUIRE(result["sources"].size() == 1);
	BOOST_REQUIRE(result["sources"][""].is_object());
}

BOOST_AUTO_TEST_CASE(standard_output_selection_wildcard_multiple_sources)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources":
		{
			"A":
			{
				"content": "pragma solidity >=0.0; contract C { function f() public pure {} }"
			},
			"B":
			{
				"content": "pragma solidity >=0.0; contract D { function f() public pure {} }"
			}
		},
		"settings":
		{
			"outputSelection":
			{
				"*": { "D": ["evm.bytecode"] }
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));

	solidity::frontend::StandardCompiler compiler;
	Json result = compiler.compile(parsedInput);

	BOOST_REQUIRE(result["contracts"].is_object());
	BOOST_REQUIRE(result["contracts"].size() == 1);
	BOOST_REQUIRE(result["contracts"]["B"].is_object());
	BOOST_REQUIRE(result["contracts"]["B"].size() == 1);
	BOOST_REQUIRE(result["contracts"]["B"]["D"].is_object());
	BOOST_REQUIRE(result["contracts"]["B"]["D"]["evm"].is_object());
	BOOST_REQUIRE(result["contracts"]["B"]["D"]["evm"]["bytecode"].is_object());
	BOOST_REQUIRE(result["sources"].is_object());
	BOOST_REQUIRE(result["sources"].size() == 2);
	BOOST_REQUIRE(result["sources"]["A"].is_object());
	BOOST_REQUIRE(result["sources"]["B"].is_object());
}

BOOST_AUTO_TEST_CASE(stopAfter_invalid_value)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources":
		{ "": { "content": "pragma solidity >=0.0; contract C { function f() public pure {} }" } },
		"settings":
		{
			"stopAfter": "rrr",
			"outputSelection":
			{
				"*": { "C": ["evm.bytecode"] }
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "Invalid value for \"settings.stopAfter\". Only valid value is \"parsing\"."));
}

BOOST_AUTO_TEST_CASE(stopAfter_invalid_type)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources":
		{ "": { "content": "pragma solidity >=0.0; contract C { function f() public pure {} }" } },
		"settings":
		{
			"stopAfter": 3,
			"outputSelection":
			{
				"*": { "C": ["evm.bytecode"] }
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "\"settings.stopAfter\" must be a string."));
}

BOOST_AUTO_TEST_CASE(stopAfter_bin_conflict)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources":
		{ "": { "content": "pragma solidity >=0.0; contract C { function f() public pure {} }" } },
		"settings":
		{
			"stopAfter": "parsing",
			"outputSelection":
			{
				"*": { "C": ["evm.bytecode"] }
			}
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(containsError(result, "JSONError", "Requested output selection conflicts with \"settings.stopAfter\"."));
}

BOOST_AUTO_TEST_CASE(stopAfter_ast_output)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"a.sol": {
				"content": "// SPDX-License-Identifier: GPL-3.0\nimport \"tes32.sol\";\n contract C is X { constructor() {} }"
			}
		},
		"settings": {
			"stopAfter": "parsing",
			"outputSelection": { "*": { "": [ "ast" ] } }
		}
	}
	)";
	Json result = compile(input);
	BOOST_CHECK(result["sources"].is_object());
	BOOST_CHECK(result["sources"]["a.sol"].is_object());
	BOOST_CHECK(result["sources"]["a.sol"]["ast"].is_object());
}

BOOST_AUTO_TEST_CASE(dependency_tracking_of_abstract_contract)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"BlockRewardAuRaBase.sol": {
				"content": " contract Sacrifice { constructor() payable {} } abstract contract BlockRewardAuRaBase { function _transferNativeReward() internal { new Sacrifice(); } function _distributeTokenRewards() internal virtual; } "
			},
			"BlockRewardAuRaCoins.sol": {
				"content": " import \"./BlockRewardAuRaBase.sol\"; contract BlockRewardAuRaCoins is BlockRewardAuRaBase { function transferReward() public { _transferNativeReward(); } function _distributeTokenRewards() internal override {} } "
			}
		},
		"settings": {
			"outputSelection": {
				"BlockRewardAuRaCoins.sol": {
					"BlockRewardAuRaCoins": ["ir", "evm.bytecode.sourceMap"]
				}
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));

	solidity::frontend::StandardCompiler compiler;
	Json result = compiler.compile(parsedInput);

	BOOST_REQUIRE(result["contracts"].is_object());
	BOOST_REQUIRE(result["contracts"].size() == 1);
	BOOST_REQUIRE(result["contracts"]["BlockRewardAuRaCoins.sol"].is_object());
	BOOST_REQUIRE(result["contracts"]["BlockRewardAuRaCoins.sol"].size() == 1);
	BOOST_REQUIRE(result["contracts"]["BlockRewardAuRaCoins.sol"]["BlockRewardAuRaCoins"].is_object());
	BOOST_REQUIRE(result["contracts"]["BlockRewardAuRaCoins.sol"]["BlockRewardAuRaCoins"]["evm"].is_object());
	BOOST_REQUIRE(result["contracts"]["BlockRewardAuRaCoins.sol"]["BlockRewardAuRaCoins"]["ir"].is_string());
	BOOST_REQUIRE(result["contracts"]["BlockRewardAuRaCoins.sol"]["BlockRewardAuRaCoins"]["evm"]["bytecode"].is_object());
	BOOST_REQUIRE(result["sources"].is_object());
	BOOST_REQUIRE(result["sources"].size() == 2);
}

BOOST_AUTO_TEST_CASE(dependency_tracking_of_abstract_contract_yul)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"A.sol": {
				"content": "contract A {} contract B {} contract C { constructor() { new B(); } } contract D {}"
			}
		},
		"settings": {
			"outputSelection": {
				"A.sol": {
					"C": ["ir"]
				}
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));

	solidity::frontend::StandardCompiler compiler;
	Json result = compiler.compile(parsedInput);

	BOOST_REQUIRE(result["contracts"].is_object());
	BOOST_REQUIRE(result["contracts"].size() == 1);
	BOOST_REQUIRE(result["contracts"]["A.sol"].is_object());
	BOOST_REQUIRE(result["contracts"]["A.sol"].size() == 1);
	BOOST_REQUIRE(result["contracts"]["A.sol"]["C"].is_object());
	BOOST_REQUIRE(result["contracts"]["A.sol"]["C"]["ir"].is_string());

	const std::string& irCode = result["contracts"]["A.sol"]["C"]["ir"].get<std::string>();

	// Make sure C and B contracts are deployed
	BOOST_REQUIRE(irCode.find("object \"C") != std::string::npos);
	BOOST_REQUIRE(irCode.find("object \"B") != std::string::npos);

	// Make sure A and D are NOT deployed as they were not requested and are not
	// in any dependency
	BOOST_REQUIRE(irCode.find("object \"A") == std::string::npos);
	BOOST_REQUIRE(irCode.find("object \"D") == std::string::npos);


	BOOST_REQUIRE(result["sources"].is_object());
	BOOST_REQUIRE(result["sources"].size() == 1);
}

BOOST_AUTO_TEST_CASE(solcore_export_minimal_subset)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore", "solcoreOrigins"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.0;
				contract C {
					mapping(address => uint256) balances;
					uint256 total;

					function credit(address to, uint256 amount) external {
						balances[to] += amount;
						total += amount;
					}

					function sameBalance(address a, address b) external view returns (bool) {
						return balances[a] == balances[b];
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult.is_object());
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	BOOST_REQUIRE(contractResult["solcoreOrigins"].is_object());

	Json const& solcore = contractResult["solcore"];
	BOOST_CHECK_EQUAL(solcore["schemaVersion"].get<std::string>(), "0.1.0");
	BOOST_CHECK_EQUAL(solcore["compilerVersion"].get<std::string>(), VersionString);
	BOOST_CHECK_EQUAL(solcore["crate_name"].get<std::string>(), "C");
	BOOST_CHECK_EQUAL(solcore["exporterFamily"].get<std::string>(), "solcore-solidity-0.8");
	BOOST_REQUIRE(solcore["type_decls"].is_array());
	BOOST_REQUIRE(solcore["functions"].is_array());
	BOOST_REQUIRE(solcore["functions"].size() == 2);

	Json const& storageDecl = solcore["type_decls"][0];
	BOOST_CHECK_EQUAL(storageDecl["name"].get<std::string>(), "Storage");
	BOOST_REQUIRE(storageDecl["fields"].is_array());
	BOOST_REQUIRE(storageDecl["fields"].size() == 2);
	BOOST_CHECK_EQUAL(storageDecl["fields"][0]["name"].get<std::string>(), "balances");
	BOOST_CHECK_EQUAL(storageDecl["fields"][1]["name"].get<std::string>(), "total");

	Json const& credit = solcore["functions"][0];
	BOOST_CHECK_EQUAL(credit["name"].get<std::string>(), "credit");
	BOOST_CHECK_EQUAL(credit["return"].get<std::string>(), "unit");
	BOOST_REQUIRE(credit["body"]["kind"] == "block");
	BOOST_REQUIRE(credit["body"]["statements"].is_array());
	BOOST_REQUIRE(credit["body"]["statements"].size() == 3);
	BOOST_CHECK_EQUAL(credit["body"]["statements"][0]["kind"].get<std::string>(), "storage_map_set");
	BOOST_CHECK_EQUAL(credit["body"]["statements"][1]["kind"].get<std::string>(), "storage_set");
	BOOST_CHECK_EQUAL(credit["body"]["statements"][2]["kind"].get<std::string>(), "return");

	Json const& sameBalance = solcore["functions"][1];
	BOOST_CHECK_EQUAL(sameBalance["name"].get<std::string>(), "sameBalance");
	BOOST_CHECK_EQUAL(sameBalance["return"].get<std::string>(), "bool");
	BOOST_CHECK_EQUAL(sameBalance["body"]["statements"][0]["kind"].get<std::string>(), "return");
	BOOST_CHECK_EQUAL(
		sameBalance["body"]["statements"][0]["value"]["kind"].get<std::string>(),
		"u256_eq"
	);

	Json const& origins = contractResult["solcoreOrigins"];
	BOOST_CHECK_EQUAL(origins["schemaVersion"].get<std::string>(), "0.1.0");
	BOOST_REQUIRE(origins["entries"].is_array());
	BOOST_REQUIRE(origins["entries"].size() == 4);
	BOOST_CHECK_EQUAL(origins["entries"][0]["originId"].get<std::string>(), "state:balances");
}

// --- Internal function used as a value: bounded defunctionalization tests ---
// (SOLCORE_FNPTR_DEFUNCTIONALIZATION_DESIGN §5.A). Looks up a JSON entry by
// its exported "name" in a `solcore["functions"]` or
// `solcore["internal_functions"]` array — the same lookup idiom already
// used ad hoc (over "originalName") by
// solcore_export_overloaded_internal_function_keeps_nested_mapping_write,
// above.
Json const* findExportedFunction(Json const& _functionArray, std::string const& _name)
{
	for (auto const& fn: _functionArray)
		if (fn.value("name", ""s) == _name)
			return &fn;
	return nullptr;
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_direct_literal_argument_is_specialized)
{
	// T1 (positive/singleton): a receiver taking an internal-function-typed
	// parameter, called from two sites with two different direct-literal
	// targets. Both call sites must get REAL bodies calling distinct
	// specialized siblings with the function-typed parameter erased; the
	// GENERIC (unspecialized) receiver must keep failing closed at the now
	// precise indirect-call message (not the old "used as a value" one).
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract FnPtrRepro {
					function bump(uint256 x) external pure returns (uint256) {
						return _apply(_add, x);
					}
					function drop(uint256 x) external pure returns (uint256) {
						return _apply(_subtract, x);
					}
					function _apply(function(uint256) internal pure returns (uint256) op, uint256 x) private pure returns (uint256) {
						return op(x);
					}
					function _add(uint256 a) private pure returns (uint256) {
						return a + 1;
					}
					function _subtract(uint256 a) private pure returns (uint256) {
						return a - 1;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "FnPtrRepro");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	// solcoreVersion is deliberately left unbumped by this feature (see the
	// comment on its assignment in exportContract) so it stays a reliable
	// "nothing else in this artifact changed" signal for the corpus diff.
	BOOST_CHECK_EQUAL(solcore["solcoreVersion"].get<std::string>(), "0.2.0");

	Json const* bump = findExportedFunction(solcore["functions"], "bump");
	Json const* drop = findExportedFunction(solcore["functions"], "drop");
	BOOST_REQUIRE(bump != nullptr);
	BOOST_REQUIRE(drop != nullptr);
	BOOST_CHECK_MESSAGE(
		bump->at("body").dump().find("\"kind\":\"unsupported_body\"") == std::string::npos,
		"bump() must get a real (non-unsupported) body once its fn-ptr argument is specialized");
	BOOST_CHECK_MESSAGE(
		drop->at("body").dump().find("\"kind\":\"unsupported_body\"") == std::string::npos,
		"drop() must get a real (non-unsupported) body once its fn-ptr argument is specialized");

	std::string bumpCallee = bump->at("body")["statements"][0]["value"]["function"].get<std::string>();
	std::string dropCallee = drop->at("body")["statements"][0]["value"]["function"].get<std::string>();
	BOOST_CHECK_NE(bumpCallee, dropCallee);
	BOOST_CHECK_NE(bumpCallee.find("_apply__fnptr__op__"), std::string::npos);
	BOOST_CHECK_NE(dropCallee.find("_apply__fnptr__op__"), std::string::npos);

	Json const* addSibling = findExportedFunction(solcore["internal_functions"], bumpCallee);
	Json const* subSibling = findExportedFunction(solcore["internal_functions"], dropCallee);
	BOOST_REQUIRE(addSibling != nullptr);
	BOOST_REQUIRE(subSibling != nullptr);
	// The function-typed parameter must be erased from the specialized
	// signature: original _apply has 2 params (op, x), the siblings have 1.
	BOOST_REQUIRE(addSibling->at("params").is_array());
	BOOST_CHECK_EQUAL(addSibling->at("params").size(), 1u);
	BOOST_CHECK_EQUAL(addSibling->at("params")[0]["name"].get<std::string>(), "x");
	BOOST_CHECK_EQUAL(
		addSibling->at("body")["statements"][0]["value"]["function"].get<std::string>(), "_add");
	BOOST_CHECK_EQUAL(
		subSibling->at("body")["statements"][0]["value"]["function"].get<std::string>(), "_subtract");
	BOOST_CHECK_EQUAL(addSibling->at("fnptr_specialization")["of"].get<std::string>(), "_apply");
	BOOST_REQUIRE(addSibling->at("ast_write_oracle").is_object());

	// The GENERIC (unspecialized) _apply is still exported (append-only
	// artifact policy) but now fails closed at the NEW precise indirect-call
	// message rather than the old blanket "used as a value" one.
	Json const* genericApply = findExportedFunction(solcore["internal_functions"], "_apply");
	BOOST_REQUIRE(genericApply != nullptr);
	BOOST_CHECK_EQUAL(genericApply->at("body")["kind"].get<std::string>(), "unsupported_body");
	BOOST_CHECK_MESSAGE(
		genericApply->at("body")["error"].get<std::string>().find("indirect call through an internal function value") != std::string::npos,
		"the generic _apply must fail at the new precise indirect-call message, not the old blanket one");
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_virtual_argument_binds_derived_override)
{
	// T2 (virtual dispatch): the base passes a VIRTUAL function as a value;
	// the derived contract overrides it. Specialization must bind the
	// DERIVED override (the resolveVirtual winner against the exporting
	// contract), not the lexically-referenced base declaration — mirrors
	// virtualCallTargetName's existing rule and solc's own codegen
	// (FunctionDefinition::resolveVirtual at pointer-creation time).
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract FnPtrVirtualBase {
					function run(uint256 x) external pure returns (uint256) {
						return _apply(_hook, x);
					}
					function _apply(function(uint256) internal pure returns (uint256) op, uint256 x) internal pure returns (uint256) {
						return op(x);
					}
					function _hook(uint256 a) internal virtual pure returns (uint256) {
						return a;
					}
				}
				contract FnPtrVirtualDerived is FnPtrVirtualBase {
					function _hook(uint256 a) internal pure override returns (uint256) {
						return a + 100;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "FnPtrVirtualDerived");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];

	// Exactly one _hook implementation is reachable in this export unit (the
	// override) — confirms the pre-existing override-collapsing behavior
	// this test relies on to distinguish base vs. derived binding.
	unsigned hookCount = 0;
	Json const* hookFn = nullptr;
	for (auto const& fn: solcore["internal_functions"])
		if (fn.value("originalName", fn["name"].get<std::string>()) == "_hook")
		{
			++hookCount;
			hookFn = &fn;
		}
	BOOST_REQUIRE_EQUAL(hookCount, 1u);
	BOOST_REQUIRE(hookFn != nullptr);
	// The DERIVED override's body ("a + 100") must be what's exported under
	// this name, not the base's plain passthrough ("a").
	BOOST_CHECK_NE(hookFn->at("body").dump().find("\"kind\":\"u256_add\""), std::string::npos);

	Json const* run = findExportedFunction(solcore["functions"], "run");
	BOOST_REQUIRE(run != nullptr);
	std::string runCallee = run->at("body")["statements"][0]["value"]["function"].get<std::string>();
	Json const* sibling = findExportedFunction(solcore["internal_functions"], runCallee);
	BOOST_REQUIRE(sibling != nullptr);
	// The specialized sibling must call the SAME exported name as the
	// override entry found above (the resolveVirtual winner), proving the
	// binding resolved to the derived override.
	BOOST_CHECK_EQUAL(
		sibling->at("body")["statements"][0]["value"]["function"].get<std::string>(),
		hookFn->at("name").get<std::string>());
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_conditional_argument_fails_closed)
{
	// T3 (adversarial: conditionally-selected target). A ternary between two
	// function literals is not a single statically-known target — must fail
	// closed with the new precise message, and must NOT emit any
	// `__fnptr__` sibling.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract T3Conditional {
					function pick(bool c, uint256 x) external pure returns (uint256) {
						return _apply(c ? _add : _subtract, x);
					}
					function _apply(function(uint256) internal pure returns (uint256) op, uint256 x) private pure returns (uint256) {
						return op(x);
					}
					function _add(uint256 a) private pure returns (uint256) { return a + 1; }
					function _subtract(uint256 a) private pure returns (uint256) { return a - 1; }
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "T3Conditional");
	Json const& solcore = contractResult["solcore"];

	Json const* pick = findExportedFunction(solcore["functions"], "pick");
	BOOST_REQUIRE(pick != nullptr);
	BOOST_CHECK_EQUAL(pick->at("body")["kind"].get<std::string>(), "unsupported_body");
	BOOST_CHECK_MESSAGE(
		pick->at("body")["error"].get<std::string>().find("not a direct internal function reference") != std::string::npos,
		"a conditionally-selected fn-ptr argument must fail closed with the precise message");

	for (auto const& fn: solcore["internal_functions"])
		BOOST_CHECK_MESSAGE(
			fn["name"].get<std::string>().find("__fnptr__") == std::string::npos,
			"no specialized sibling may be emitted for a conditional argument");
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_reassigned_parameter_fails_closed)
{
	// T4 (adversarial: reassignment). The callee reassigns its own
	// function-typed parameter before calling it — the singleton-target
	// claim is FALSE for this callee, so specialization must refuse
	// (admissibility check), not silently bind to the call-site argument.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract T4Reassign {
					function bump(uint256 x) external pure returns (uint256) {
						return _apply(_add, x);
					}
					function _apply(function(uint256) internal pure returns (uint256) op, uint256 x) private pure returns (uint256) {
						op = _subtract;
						return op(x);
					}
					function _add(uint256 a) private pure returns (uint256) { return a + 1; }
					function _subtract(uint256 a) private pure returns (uint256) { return a - 1; }
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "T4Reassign");
	Json const& solcore = contractResult["solcore"];

	Json const* bump = findExportedFunction(solcore["functions"], "bump");
	BOOST_REQUIRE(bump != nullptr);
	BOOST_CHECK_EQUAL(bump->at("body")["kind"].get<std::string>(), "unsupported_body");
	BOOST_CHECK_MESSAGE(
		bump->at("body")["error"].get<std::string>().find("reassigned") != std::string::npos,
		"a callee that reassigns its fn-typed parameter must refuse via the admissibility check");
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_indirect_call_never_binds_to_same_named_function)
{
	// T5 (adversarial: name-collision regression — pins the exportExpr:3535
	// fallback fix). The contract ALSO defines a real internal function
	// literally named `op` (the fn-ptr parameter's name). The GENERIC
	// (unspecialized) _apply is always exported alongside its specialized
	// siblings; its own `op(x)` indirect call must fail closed, never
	// silently bind to the unrelated same-named function.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract T5Collision {
					function bump(uint256 x) external pure returns (uint256) {
						return _apply(_add, x);
					}
					function _apply(function(uint256) internal pure returns (uint256) op, uint256 x) private pure returns (uint256) {
						return op(x);
					}
					function _add(uint256 a) private pure returns (uint256) { return a + 1; }
					function op(uint256 y) private pure returns (uint256) { return y + 999; }
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "T5Collision");
	Json const& solcore = contractResult["solcore"];

	Json const* genericApply = findExportedFunction(solcore["internal_functions"], "_apply");
	BOOST_REQUIRE(genericApply != nullptr);
	BOOST_CHECK_EQUAL(genericApply->at("body")["kind"].get<std::string>(), "unsupported_body");
	BOOST_CHECK_MESSAGE(
		genericApply->at("body")["error"].get<std::string>().find("cannot be resolved to a static target") != std::string::npos,
		"the generic _apply's op(x) must fail closed, not silently bind to the unrelated op() function");

	// bump() must still get a real specialized body calling _add, never the
	// unrelated same-named `op` function.
	Json const* bump = findExportedFunction(solcore["functions"], "bump");
	BOOST_REQUIRE(bump != nullptr);
	BOOST_CHECK_MESSAGE(
		bump->at("body").dump().find("\"kind\":\"unsupported_body\"") == std::string::npos,
		"bump() itself must still specialize successfully");
	std::string bumpCallee = bump->at("body")["statements"][0]["value"]["function"].get<std::string>();
	Json const* sibling = findExportedFunction(solcore["internal_functions"], bumpCallee);
	BOOST_REQUIRE(sibling != nullptr);
	BOOST_CHECK_EQUAL(
		sibling->at("body")["statements"][0]["value"]["function"].get<std::string>(), "_add");
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_transitive_forwarding_and_self_recursion)
{
	// T6 (transitive forwarding + self-recursion). `_f(op)` forwards `op`
	// into `_g(op)` — the nested specialization must resolve correctly, and
	// a self-recursive `_f(op){ ...; _f(op, ...); }` must terminate via the
	// specialization memo (re-deriving the SAME specialized name on its own
	// recursive call) rather than looping forever or leaving a dangling
	// generic call.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract T6Transitive {
					function bump(uint256 x) external pure returns (uint256) {
						return _f(_add, x);
					}
					function _f(function(uint256) internal pure returns (uint256) op, uint256 x) private pure returns (uint256) {
						return _g(op, x);
					}
					function _g(function(uint256) internal pure returns (uint256) op, uint256 x) private pure returns (uint256) {
						return op(x);
					}
					function _add(uint256 a) private pure returns (uint256) { return a + 1; }
				}
				contract T6SelfRecursive {
					function bump(uint256 x, uint256 n) external pure returns (uint256) {
						return _f(_add, x, n);
					}
					function _f(function(uint256) internal pure returns (uint256) op, uint256 x, uint256 n) private pure returns (uint256) {
						if (n == 0) return x;
						return _f(op, op(x), n - 1);
					}
					function _add(uint256 a) private pure returns (uint256) { return a + 1; }
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	{
		Json contractResult = getContractResult(result, "fileA", "T6Transitive");
		Json const& solcore = contractResult["solcore"];
		Json const* bump = findExportedFunction(solcore["functions"], "bump");
		BOOST_REQUIRE(bump != nullptr);
		BOOST_CHECK_MESSAGE(
			bump->at("body").dump().find("\"kind\":\"unsupported_body\"") == std::string::npos,
			"bump() must specialize through the _f -> _g forwarding chain");
		std::string fSibling = bump->at("body")["statements"][0]["value"]["function"].get<std::string>();
		Json const* fFn = findExportedFunction(solcore["internal_functions"], fSibling);
		BOOST_REQUIRE(fFn != nullptr);
		std::string gCallee = fFn->at("body")["statements"][0]["value"]["function"].get<std::string>();
		Json const* gFn = findExportedFunction(solcore["internal_functions"], gCallee);
		BOOST_REQUIRE_MESSAGE(gFn != nullptr, "the transitively-forwarded _g specialization must be emitted");
		BOOST_CHECK_EQUAL(
			gFn->at("body")["statements"][0]["value"]["function"].get<std::string>(), "_add");
	}

	{
		Json contractResult = getContractResult(result, "fileA", "T6SelfRecursive");
		Json const& solcore = contractResult["solcore"];
		Json const* bump = findExportedFunction(solcore["functions"], "bump");
		BOOST_REQUIRE(bump != nullptr);
		BOOST_CHECK_MESSAGE(
			bump->at("body").dump().find("\"kind\":\"unsupported_body\"") == std::string::npos,
			"bump() must specialize the self-recursive _f");
		std::string fSibling = bump->at("body")["statements"][0]["value"]["function"].get<std::string>();
		// Exactly ONE specialized sibling of _f may exist (the memo must
		// collapse the recursive call onto the SAME name, not diverge into
		// an unbounded chain of siblings).
		unsigned siblingCount = 0;
		for (auto const& fn: solcore["internal_functions"])
			if (fn["name"].get<std::string>().find("_f__fnptr__") == 0)
				++siblingCount;
		BOOST_CHECK_EQUAL(siblingCount, 1u);
		std::string dump = findExportedFunction(solcore["internal_functions"], fSibling)->at("body").dump();
		// The recursive call inside the specialized body must target ITSELF
		// (the same specialized name), and the inner op(x) call must
		// resolve directly to _add.
		BOOST_CHECK_NE(dump.find("\"function\":\"" + fSibling + "\""), std::string::npos);
		BOOST_CHECK_NE(dump.find("\"function\":\"_add\""), std::string::npos);
	}
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_statement_position_indirect_call_fails_closed)
{
	// T7 (statement-position indirect call). `op(x);` used as a bare
	// statement (return value discarded) reaches the generic FunctionCall
	// fallback via exportStmt's ultimate `exportExpr(expr)` fallback rather
	// than exportStmt's own dedicated internal-call branch — confirms that
	// single fallback fix covers both the expression and statement paths.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract T7Stmt {
					function bump(uint256 x) external pure {
						_apply(_add, x);
					}
					function _apply(function(uint256) internal pure returns (uint256) op, uint256 x) private pure {
						op(x);
					}
					function _add(uint256 a) private pure returns (uint256) { return a + 1; }
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "T7Stmt");
	Json const& solcore = contractResult["solcore"];

	Json const* bump = findExportedFunction(solcore["functions"], "bump");
	BOOST_REQUIRE(bump != nullptr);
	BOOST_CHECK_MESSAGE(
		bump->at("body").dump().find("\"kind\":\"unsupported_body\"") == std::string::npos,
		"bump() must specialize even though _apply's own op(x) call is statement-position");

	std::string sibling = bump->at("body")["statements"][0]["value"]["function"].get<std::string>();
	Json const* siblingFn = findExportedFunction(solcore["internal_functions"], sibling);
	BOOST_REQUIRE(siblingFn != nullptr);
	BOOST_CHECK_MESSAGE(
		siblingFn->at("body").dump().find("\"kind\":\"unsupported_body\"") == std::string::npos,
		"the specialized sibling's statement-position op(x) call must resolve, not fail closed");
	BOOST_CHECK_NE(siblingFn->at("body").dump().find("\"function\":\"_add\""), std::string::npos);
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_named_argument_call_fails_closed)
{
	// T8 (adversarial: named-argument call). Positional binding is required;
	// a named-argument call site must refuse rather than guess which named
	// argument lands on the function-typed parameter.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract T8Named {
					function bump(uint256 x) external pure returns (uint256) {
						return _apply({op: _add, x: x});
					}
					function _apply(function(uint256) internal pure returns (uint256) op, uint256 x) private pure returns (uint256) {
						return op(x);
					}
					function _add(uint256 a) private pure returns (uint256) { return a + 1; }
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "T8Named");
	Json const& solcore = contractResult["solcore"];

	Json const* bump = findExportedFunction(solcore["functions"], "bump");
	BOOST_REQUIRE(bump != nullptr);
	BOOST_CHECK_EQUAL(bump->at("body")["kind"].get<std::string>(), "unsupported_body");
	BOOST_CHECK_MESSAGE(
		bump->at("body")["error"].get<std::string>().find("named-argument call") != std::string::npos,
		"a named-argument call binding a fn-ptr parameter must fail closed with the precise message");
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_specialized_name_collision_fails_closed)
{
	// T9 (adversarial: a USER-DEFINED internal function is literally named
	// like the synthesized specialization). Emitting the specialization
	// under that name would merge two distinct definitions; SKIPPING it
	// would silently bind bump()'s rewritten call site to the unrelated
	// decoy (same arity!) — a silent mis-bind. The only sound outcome is a
	// loud failure of the whole contract export: the throw is absorbed by
	// the whole-contract unsupportedExport wrapper into a structured
	// `{"unsupported": true, "reason": ...}` stub naming the collision.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract T9Collision {
					function bump(uint256 x) external pure returns (uint256) {
						return _apply(_add, x);
					}
					function _apply(function(uint256) internal pure returns (uint256) op, uint256 x) private pure returns (uint256) {
						return op(x);
					}
					function _add(uint256 a) private pure returns (uint256) { return a + 1; }
					// Decoy: exactly the name the specialization would get, same
					// arity as the rewritten call site.
					function _apply__fnptr__op___add(uint256 y) private pure returns (uint256) { return 666; }
					function decoyKeepAlive(uint256 y) external pure returns (uint256) {
						return _apply__fnptr__op___add(y);
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "T9Collision");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_REQUIRE_MESSAGE(
		solcore.value("unsupported", false),
		"a specialization-name collision with a user-defined function must fail the whole "
		"contract export closed, never silently skip or merge the specialization");
	std::string reason = solcore.value("reason", std::string{});
	BOOST_CHECK_MESSAGE(
		reason.find("specialization name") != std::string::npos &&
			reason.find("collides") != std::string::npos,
		"the structured rejection must name the colliding specialization; got: " + reason);
	// Regression pin: no partial artifact — a mis-bound bump() must never be
	// emitted alongside (or instead of) the refused specialization.
	BOOST_CHECK(!solcore.contains("functions"));
}

BOOST_AUTO_TEST_CASE(solcore_export_fnptr_struct_member_indirect_call_fails_closed)
{
	// T10 (adversarial: function pointer stored in a struct field, called
	// through MEMBER access). The identifier-callee guard cannot see this
	// shape; before the member-access guard, `s.f(x, 1)` name-punted to a
	// bare internal_call "f" — mis-binding to the unrelated real internal
	// function `f` (multiply) below. Both the store (`arm`) and the call
	// (`callIt`) must fail closed.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;
				contract T10Member {
					struct S { function(uint256, uint256) internal pure returns (uint256) f; }
					S internal s;
					uint256 public acc;
					function _add(uint256 a, uint256 b) internal pure returns (uint256) { return a + b; }
					function f(uint256 a, uint256 b) internal pure returns (uint256) { return a * b; }
					function keepAlive(uint256 a) external pure returns (uint256) { return f(a, 2); }
					function arm() external { s.f = _add; }
					function callIt(uint256 x) external returns (uint256) { acc = s.f(x, 1); return acc; }
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "T10Member");
	Json const& solcore = contractResult["solcore"];

	Json const* callIt = findExportedFunction(solcore["functions"], "callIt");
	BOOST_REQUIRE(callIt != nullptr);
	BOOST_CHECK_EQUAL(callIt->at("body")["kind"].get<std::string>(), "unsupported_body");
	BOOST_CHECK_MESSAGE(
		callIt->at("body")["error"].get<std::string>().find("member access") != std::string::npos,
		"an indirect call through a struct-member function pointer must fail closed, "
		"never name-punt to the bare member name");
	// Regression pin: the old behavior emitted internal_call "f" here, which
	// would have silently bound to the real (multiplying) internal `f`.
	BOOST_CHECK_EQUAL(callIt->at("body").dump().find("\"function\":\"f\""), std::string::npos);

	Json const* arm = findExportedFunction(solcore["functions"], "arm");
	BOOST_REQUIRE(arm != nullptr);
	BOOST_CHECK_EQUAL(arm->at("body")["kind"].get<std::string>(), "unsupported_body");
}

BOOST_AUTO_TEST_CASE(solcore_export_tags_verified_oz_checkpoints_queries)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"@openzeppelin/contracts/utils/structs/Checkpoints.sol", R"(
				pragma solidity >=0.8.20;
				library Checkpoints {
					struct Trace256 {
						uint256 value;
					}

					function upperLookup(Trace256 storage self, uint256) internal view returns (uint256) {
						return self.value;
					}

					struct Trace224 {
						uint224 value;
					}

					function lowerLookup(Trace224 storage self, uint32) internal view returns (uint224) {
						return self.value;
					}

					function upperLookup(Trace224 storage self, uint32) internal view returns (uint224) {
						return self.value;
					}

					function upperLookupRecent(Trace224 storage self, uint32) internal view returns (uint224) {
						return self.value;
					}

					function latest(Trace224 storage self) internal view returns (uint224) {
						return self.value;
					}

					function length(Trace224 storage self) internal view returns (uint256) {
						return self.value;
					}
				}
			)"},
			{"@openzeppelin/contracts/utils/Checkpoints.sol", R"(
				pragma solidity >=0.8.20;
				library Checkpoints {
					struct History {
						uint256 value;
					}

					function latest(History storage self) internal view returns (uint256) {
						return self.value;
					}
				}
			)"},
			{"counterfeit/Checkpoints.sol", R"(
				pragma solidity >=0.8.20;
				library Checkpoints {
					struct Trace224 {
						uint224 value;
					}

					function upperLookup(Trace224 storage self, uint32) internal view returns (uint224) {
						return self.value;
					}
				}
			)"},
			{"fileA", R"(
				pragma solidity >=0.8.20;
				import {Checkpoints} from "@openzeppelin/contracts/utils/structs/Checkpoints.sol";
				import {Checkpoints as LegacyCheckpoints} from "@openzeppelin/contracts/utils/Checkpoints.sol";
				import {Checkpoints as CounterfeitCheckpoints} from "counterfeit/Checkpoints.sol";

				contract C {
					using Checkpoints for Checkpoints.Trace224;
					using Checkpoints for Checkpoints.Trace256;
					using CounterfeitCheckpoints for CounterfeitCheckpoints.Trace224;

					Checkpoints.Trace224 private trusted;
					Checkpoints.Trace256 private trusted256;
					CounterfeitCheckpoints.Trace224 private counterfeit;
					LegacyCheckpoints.History private legacy;

					function extension(uint32 key) external view returns (uint224) {
						return trusted.upperLookup(key);
					}

					function qualified(uint32 key) external view returns (uint224) {
						return Checkpoints.upperLookup(trusted, key);
					}

					function trace256(uint256 key) external view returns (uint256) {
						return trusted256.upperLookup(key);
					}

					function lower(uint32 key) external view returns (uint224) {
						return trusted.lowerLookup(key);
					}

					function recent(uint32 key) external view returns (uint224) {
						return trusted.upperLookupRecent(key);
					}

					function latestValue() external view returns (uint224) {
						return trusted.latest();
					}

					function checkpointCount() external view returns (uint256) {
						return trusted.length();
					}

					function legacyLatest() external view returns (uint256) {
						return LegacyCheckpoints.latest(legacy);
					}

					function fake(uint32 key) external view returns (uint224) {
						return counterfeit.upperLookup(key);
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_REQUIRE(solcore["functions"].is_array());

	auto queryCall = [&solcore](std::string const& _functionName) -> Json const*
	{
		for (Json const& function: solcore["functions"])
		{
			if (
				!function.contains("name") ||
				!function["name"].is_string() ||
				function["name"].get<std::string>() != _functionName
			)
				continue;

			Json const& statements = function["body"]["statements"];
			if (
				statements.size() == 1 &&
				statements[0].contains("value") &&
				statements[0]["value"].is_object()
			)
				return &statements[0]["value"];
		}
		return nullptr;
	};
	auto checkQuery = [&queryCall](
		std::string const& _functionName,
		std::string const& _contractId
	)
	{
		Json const* call = queryCall(_functionName);
		BOOST_REQUIRE(call);
		BOOST_REQUIRE(call->contains("contractId"));
		BOOST_REQUIRE(call->contains("runtimeKind"));
		BOOST_CHECK_EQUAL(call->at("kind").get<std::string>(), "internal_call");
		BOOST_CHECK_EQUAL(call->at("contractId").get<std::string>(), _contractId);
		BOOST_CHECK_EQUAL(call->at("runtimeKind").get<std::string>(), "oz_checkpoints_query");
	};

	// `extension` and `qualified` take the distinct using-for and
	// library-qualified expression emission paths, respectively.
	checkQuery("extension", "@openzeppelin/contracts/utils/structs/Checkpoints.sol:Checkpoints");
	checkQuery("qualified", "@openzeppelin/contracts/utils/structs/Checkpoints.sol:Checkpoints");
	checkQuery("trace256", "@openzeppelin/contracts/utils/structs/Checkpoints.sol:Checkpoints");
	checkQuery("lower", "@openzeppelin/contracts/utils/structs/Checkpoints.sol:Checkpoints");
	checkQuery("recent", "@openzeppelin/contracts/utils/structs/Checkpoints.sol:Checkpoints");
	checkQuery("latestValue", "@openzeppelin/contracts/utils/structs/Checkpoints.sol:Checkpoints");
	checkQuery("checkpointCount", "@openzeppelin/contracts/utils/structs/Checkpoints.sol:Checkpoints");
	checkQuery("legacyLatest", "@openzeppelin/contracts/utils/Checkpoints.sol:Checkpoints");

	Json const* counterfeitCall = queryCall("fake");
	BOOST_REQUIRE(counterfeitCall);
	BOOST_REQUIRE(counterfeitCall->contains("contractId"));
	BOOST_CHECK_EQUAL(
		counterfeitCall->at("contractId").get<std::string>(),
		"counterfeit/Checkpoints.sol:Checkpoints"
	);
	BOOST_CHECK(!counterfeitCall->contains("runtimeKind"));
}

BOOST_AUTO_TEST_CASE(solcore_export_is_version_tagged_when_unsupported)
{
	// A `transient`-location state variable is deliberately fail-closed at
	// contract level (exporting it as ordinary persistent storage would be a
	// false model), so it exercises the whole-contract unsupportedExport
	// path.  (The original fixture used `cond && cond`, but boolean
	// conjunction has long since gained faithful `bool_and` lowering.)
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.28;
				contract C {
					uint256 transient temp;
					function bump() external returns (uint256) {
						temp += 1;
						return temp;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	BOOST_CHECK(contractResult["solcore"]["unsupported"].get<bool>());
	// The structured rejection must say precisely why, not just flag failure.
	BOOST_CHECK(
		contractResult["solcore"]["reason"].get<std::string>().find("transient") != std::string::npos);
	BOOST_CHECK_EQUAL(contractResult["solcore"]["compilerVersion"].get<std::string>(), VersionString);
	BOOST_CHECK_EQUAL(contractResult["solcore"]["exporterFamily"].get<std::string>(), "solcore-solidity-0.8");
}

BOOST_AUTO_TEST_CASE(solcore_export_abi_decode_type_list_and_method_selector)
{
	// Regression pin for the Reserve-protocol corpus generation failures
	// (PermitLib / SignatureCheckerUpgradeable / SafeERC20 /
	// SafeERC20Upgradeable, plus OZ ERC4626's _tryGetAssetDecimals):
	//
	// 1. abi.decode(data, (T1, ...)) — the parenthesized second argument is a
	//    syntactic *type list* (ElementaryTypeNameExpression components inside
	//    a TupleExpression), not a value expression. exportExpr used to fall
	//    through every recognized Expression subtype on it and substitute a
	//    {"kind":"u256","value":"0","_unsupported_expr":true} placeholder,
	//    corrupting the call's semantics and failing the whole contract
	//    downstream (the OCaml frontend fail-louds on the sentinel). It must
	//    now be exported out-of-band as canonical type-name strings in a
	//    sibling `decode_types` field, with only the data argument in `args`.
	//
	// 2. Interface.method.selector — the base of the member access is a
	//    function, not a value; value-lowering it used to throw and end in the
	//    same placeholder. It must now be exported as a `method_selector` node
	//    with authoritative selector_hex/method_signature read from the type
	//    annotations.
	//
	// The probe bodies below mirror the exact corpus expression shapes
	// (SafeERC20._callOptionalReturn's `abi.decode(returndata, (bool))` and
	// SignatureCheckerUpgradeable.isValidERC1271SignatureNow's
	// `abi.decode(result, (bytes32)) == bytes32(IERC1271.isValidSignature.selector)`).
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.0;
				interface IERC1271Like {
					function isValidSignature(bytes32 hash, bytes memory signature) external view returns (bytes4);
				}
				library ProbeLib {
					function probeBool(bytes memory returndata) internal pure returns (bool) {
						return returndata.length == 0 || abi.decode(returndata, (bool));
					}
					function probeSelectorWord(bytes memory result) internal pure returns (bool) {
						return abi.decode(result, (bytes32)) == bytes32(IERC1271Like.isValidSignature.selector);
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "ProbeLib");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];

	// Fail-closed: nothing in this artifact may be a silent 0-placeholder or
	// an unexported body — both probes must lower faithfully.
	std::string serialized = solcore.dump();
	BOOST_CHECK(serialized.find("_unsupported_expr") == std::string::npos);
	BOOST_CHECK(serialized.find("unsupported_body") == std::string::npos);

	BOOST_REQUIRE(solcore["internal_functions"].is_array());
	BOOST_REQUIRE(solcore["internal_functions"].size() == 2);

	Json const& probeBool = solcore["internal_functions"][0];
	BOOST_CHECK_EQUAL(probeBool["name"].get<std::string>(), "probeBool");
	// return returndata.length == 0 || abi.decode(returndata, (bool));
	Json const& decodeBool = probeBool["body"]["statements"][0]["value"]["rhs"];
	BOOST_CHECK_EQUAL(decodeBool["kind"].get<std::string>(), "internal_call");
	BOOST_CHECK_EQUAL(decodeBool["function"].get<std::string>(), "abi_decode");
	BOOST_REQUIRE(decodeBool["args"].is_array());
	// Only the data argument — the type list must NOT appear as a value arg.
	BOOST_REQUIRE(decodeBool["args"].size() == 1);
	BOOST_CHECK_EQUAL(decodeBool["args"][0]["kind"].get<std::string>(), "local");
	BOOST_REQUIRE(decodeBool["decode_types"].is_array());
	BOOST_REQUIRE(decodeBool["decode_types"].size() == 1);
	BOOST_CHECK_EQUAL(decodeBool["decode_types"][0].get<std::string>(), "bool");

	Json const& probeSelector = solcore["internal_functions"][1];
	BOOST_CHECK_EQUAL(probeSelector["name"].get<std::string>(), "probeSelectorWord");
	// return abi.decode(result, (bytes32)) == bytes32(IERC1271Like.isValidSignature.selector);
	Json const& comparison = probeSelector["body"]["statements"][0]["value"];
	BOOST_CHECK_EQUAL(comparison["kind"].get<std::string>(), "u256_eq");
	Json const& decodeWord = comparison["lhs"];
	BOOST_CHECK_EQUAL(decodeWord["function"].get<std::string>(), "abi_decode");
	BOOST_REQUIRE(decodeWord["decode_types"].is_array());
	BOOST_REQUIRE(decodeWord["decode_types"].size() == 1);
	BOOST_CHECK_EQUAL(decodeWord["decode_types"][0].get<std::string>(), "bytes32");
	Json const& selector = comparison["rhs"];
	BOOST_CHECK_EQUAL(selector["kind"].get<std::string>(), "method_selector");
	BOOST_CHECK_EQUAL(selector["method_name"].get<std::string>(), "isValidSignature");
	BOOST_CHECK_EQUAL(selector["method_signature"].get<std::string>(), "isValidSignature(bytes32,bytes)");
	BOOST_CHECK_EQUAL(selector["selector_hex"].get<std::string>(), "1626ba7e");
	BOOST_CHECK_EQUAL(selector["contractId"].get<std::string>(), "IERC1271Like");
}

BOOST_AUTO_TEST_CASE(solcore_export_namespaced_storage)
{
	// ERC-7201 namespaced storage pattern: _getTokenStorage() returns a storage
	// pointer to a struct at a computed slot. The exporter should flatten the
	// sub-storage fields into the main Storage type and emit storage_get/set_storage
	// operations instead of local struct manipulation.
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract C {
					uint256 public totalMinted;

					struct TokenStorage {
						mapping(address => uint256) balances;
						uint256 totalSupply;
					}

					bytes32 private constant TOKEN_STORAGE_LOCATION =
						0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd00;

					function _getTokenStorage() private pure returns (TokenStorage storage $) {
						assembly {
							$.slot := TOKEN_STORAGE_LOCATION
						}
					}

					function mint(address account, uint256 amount) external {
						totalMinted += amount;
						TokenStorage storage $ = _getTokenStorage();
						$.totalSupply += amount;
						$.balances[account] += amount;
					}

					function balanceOf(address account) external view returns (uint256) {
						return _getTokenStorage().balances[account];
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];

	// 1. Storage type should include flattened sub-storage fields
	Json const& storageDecl = solcore["type_decls"][0];
	BOOST_CHECK_EQUAL(storageDecl["name"].get<std::string>(), "Storage");
	BOOST_REQUIRE(storageDecl["fields"].is_array());

	// Should have: totalMinted (direct) + balances, totalSupply (from TokenStorage)
	bool hasBalances = false, hasTotalSupply = false, hasTotalMinted = false;
	for (auto const& field : storageDecl["fields"])
	{
		std::string name = field["name"].get<std::string>();
		if (name == "totalMinted") hasTotalMinted = true;
		if (name == "token_balances" || name == "balances") hasBalances = true;
		if (name == "token_totalSupply" || name == "totalSupply") hasTotalSupply = true;
	}
	BOOST_CHECK_MESSAGE(hasTotalMinted, "Storage should have totalMinted field");
	BOOST_CHECK_MESSAGE(hasBalances, "Storage should have flattened balances field from TokenStorage");
	BOOST_CHECK_MESSAGE(hasTotalSupply, "Storage should have flattened totalSupply field from TokenStorage");

	// 2. _getTokenStorage should NOT appear as an internal function
	bool hasGetterFunction = false;
	for (auto const& fn : solcore["internal_functions"])
	{
		if (fn["name"].get<std::string>() == "_getTokenStorage")
			hasGetterFunction = true;
	}
	BOOST_CHECK_MESSAGE(!hasGetterFunction,
		"_getTokenStorage should be eliminated, not exported as internal function");

	// 3. mint function should use storage_get/set_storage, not local struct ops
	Json const* mintFn = nullptr;
	for (auto const& fn : solcore["functions"])
	{
		if (fn["name"].get<std::string>() == "mint")
			mintFn = &fn;
	}
	BOOST_REQUIRE_MESSAGE(mintFn != nullptr, "mint function should exist");

	// Check that mint body contains storage_set or storage_map_set operations
	// (not struct_update or array_get on a local variable)
	std::string mintBody = mintFn->dump();
	BOOST_CHECK_MESSAGE(
		mintBody.find("\"storage_set\"") != std::string::npos ||
		mintBody.find("\"set_storage\"") != std::string::npos,
		"mint should emit set_storage for totalSupply write, not struct_update");
	BOOST_CHECK_MESSAGE(
		mintBody.find("\"storage_map_set\"") != std::string::npos,
		"mint should emit storage_map_set for balances write, not array_set_expr");

	// 4. balanceOf should use storage_map_get, not internal_call + array_get
	Json const* balanceOfFn = nullptr;
	for (auto const& fn : solcore["functions"])
	{
		if (fn["name"].get<std::string>() == "balanceOf")
			balanceOfFn = &fn;
	}
	BOOST_REQUIRE_MESSAGE(balanceOfFn != nullptr, "balanceOf function should exist");

	std::string balanceOfBody = balanceOfFn->dump();
	BOOST_CHECK_MESSAGE(
		balanceOfBody.find("\"storage_map_get\"") != std::string::npos,
		"balanceOf should emit storage_map_get for balances read, not internal_call");
BOOST_CHECK_MESSAGE(
	balanceOfBody.find("\"internal_call\"") == std::string::npos ||
	balanceOfBody.find("\"_getTokenStorage\"") == std::string::npos,
	"balanceOf should not call _getTokenStorage");
}

BOOST_AUTO_TEST_CASE(solcore_export_overloaded_sub_storage_getters_are_disambiguated)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract C {
					struct TokenStorage {
						mapping(address => uint256) balances;
						uint256 totalSupply;
					}

					struct FeeStorage {
						uint256 fee;
					}

					bytes32 private constant TOKEN_STORAGE_LOCATION =
						0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd00;
					bytes32 private constant FEE_STORAGE_LOCATION =
						0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd01;

					function _getStorage(uint256) private pure returns (TokenStorage storage slot_) {
						assembly {
							slot_.slot := TOKEN_STORAGE_LOCATION
						}
					}

					function _getStorage(address) private pure returns (FeeStorage storage slot_) {
						assembly {
							slot_.slot := FEE_STORAGE_LOCATION
						}
					}

					function balanceOf(address account) external view returns (uint256) {
						return _getStorage(uint256(0)).balances[account];
					}

					function fee() external view returns (uint256) {
						return _getStorage(address(0)).fee;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_REQUIRE(solcore["functions"].is_array());
	BOOST_REQUIRE(solcore["internal_functions"].is_array());
	BOOST_REQUIRE(solcore["sub_storage_getters"].is_array());
	BOOST_CHECK_EQUAL(solcore["sub_storage_getters"].size(), 0);
	BOOST_CHECK_EQUAL(solcore["internal_functions"].size(), 0);

	Json const* balanceOfFn = nullptr;
	Json const* feeFn = nullptr;
	for (auto const& fn : solcore["functions"])
	{
		if (fn["name"].get<std::string>() == "balanceOf")
			balanceOfFn = &fn;
		else if (fn["name"].get<std::string>() == "fee")
			feeFn = &fn;
	}

	BOOST_REQUIRE_MESSAGE(balanceOfFn != nullptr, "balanceOf function should exist");
	BOOST_REQUIRE_MESSAGE(feeFn != nullptr, "fee function should exist");

	std::string balanceOfBody = balanceOfFn->dump();
	std::string feeBody = feeFn->dump();
	BOOST_CHECK_MESSAGE(
		balanceOfBody.find("\"kind\":\"storage_map_get\"") != std::string::npos,
		"overloaded token getter should lower to storage_map_get");
	BOOST_CHECK_MESSAGE(
		balanceOfBody.find("\"field\":\"token_balances\"") != std::string::npos,
		"overloaded token getter should preserve the token_ prefix");
	BOOST_CHECK_MESSAGE(
		feeBody.find("\"kind\":\"storage_get\"") != std::string::npos,
		"overloaded fee getter should lower to storage_get");
	BOOST_CHECK_MESSAGE(
		feeBody.find("\"field\":\"fee_fee\"") != std::string::npos,
		"overloaded fee getter should preserve the fee_ prefix");
}

BOOST_AUTO_TEST_CASE(solcore_export_nested_mapping_assignment)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract C {
					mapping(address => mapping(address => uint256)) public allowance;

					function _approve(address owner, address spender, uint256 value) internal {
						allowance[owner][spender] = value;
					}

					function approve(address spender, uint256 value) external returns (bool) {
						_approve(msg.sender, spender, value);
						return true;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];

	Json const* approveFn = nullptr;
	for (auto const& fn : solcore["internal_functions"])
	{
		if (fn["name"].get<std::string>() == "_approve")
			approveFn = &fn;
	}
	BOOST_REQUIRE_MESSAGE(approveFn != nullptr, "_approve internal function should exist");

	std::string approveBody = approveFn->dump();
	BOOST_CHECK_MESSAGE(
		approveBody.find("\"kind\":\"storage_map_set\"") != std::string::npos,
		"_approve should emit storage_map_set for nested mapping writes");
	BOOST_CHECK_MESSAGE(
		approveBody.find("\"field\":\"allowance\"") != std::string::npos,
		"_approve should update the allowance storage field");
	BOOST_CHECK_MESSAGE(
		approveBody.find("\"function\":\"array_set_expr\"") != std::string::npos,
		"_approve should build the updated inner mapping with array_set_expr");
	BOOST_CHECK_MESSAGE(
		approveBody.find("\"kind\":\"expr\",\"value\":{\"kind\":\"internal_call\",\"function\":\"array_set_expr\"") ==
			std::string::npos,
		"_approve should not degrade nested mapping writes to a top-level expr array_set_expr");
}

BOOST_AUTO_TEST_CASE(solcore_export_overloaded_internal_function_keeps_nested_mapping_write)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract C {
					mapping(address => mapping(address => uint256)) public allowance;

					function _approve(address owner, address spender, uint256 value) internal {
						_approve(owner, spender, value, true);
					}

					function _approve(address owner, address spender, uint256 value, bool emitEvent) internal {
						allowance[owner][spender] = value;
						if (emitEvent) {}
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_REQUIRE(solcore["internal_functions"].is_array());

	Json const* approve3 = nullptr;
	Json const* approve4 = nullptr;
	unsigned approveCount = 0;
	for (auto const& fn : solcore["internal_functions"])
	{
		if (fn.value("originalName", fn["name"].get<std::string>()) != "_approve")
			continue;
		++approveCount;
		if (fn["params"].size() == 3)
			approve3 = &fn;
		else if (fn["params"].size() == 4)
			approve4 = &fn;
	}

	BOOST_CHECK_EQUAL(approveCount, 2);
	BOOST_REQUIRE_MESSAGE(approve3 != nullptr, "three-argument _approve overload should be exported");
	BOOST_REQUIRE_MESSAGE(approve4 != nullptr, "four-argument _approve overload should be exported");
	BOOST_CHECK_NE(approve3->at("name").get<std::string>(), approve4->at("name").get<std::string>());

	std::string wrapperBody = approve3->dump();
	std::string writerBody = approve4->dump();
	BOOST_CHECK_MESSAGE(
		wrapperBody.find("\"function\":\"" + approve4->at("name").get<std::string>() + "\"") != std::string::npos,
		"three-argument _approve should call the exported four-argument overload");
	BOOST_CHECK_MESSAGE(
		writerBody.find("\"kind\":\"storage_map_set\"") != std::string::npos,
		"four-argument _approve should emit storage_map_set for allowance writes");
	BOOST_CHECK_MESSAGE(
		writerBody.find("\"field\":\"allowance\"") != std::string::npos,
		"four-argument _approve should update the allowance storage field");
}

BOOST_AUTO_TEST_CASE(solcore_export_mapping_post_increment_writes_storage_and_returns_old_value)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract C {
					mapping(address => uint256) public nonces;

					function _useNonce(address owner) internal returns (uint256) {
						return nonces[owner]++;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_REQUIRE(solcore["internal_functions"].is_array());

	Json const* useNonce = nullptr;
	for (auto const& fn : solcore["internal_functions"])
	{
		if (fn["name"].get<std::string>() == "_useNonce")
			useNonce = &fn;
	}
	BOOST_REQUIRE_MESSAGE(useNonce != nullptr, "_useNonce internal function should exist");

	std::string useNonceBody = useNonce->dump();
	BOOST_CHECK_MESSAGE(
		useNonceBody.find("\"kind\":\"block\"") != std::string::npos,
		"_useNonce should lower return nonces[owner]++ into an explicit block");
	BOOST_CHECK_MESSAGE(
		useNonceBody.find("\"kind\":\"let\"") != std::string::npos,
		"_useNonce should capture the old nonce value before incrementing");
	BOOST_CHECK_MESSAGE(
		useNonceBody.find("\"kind\":\"storage_map_set\"") != std::string::npos,
		"_useNonce should emit storage_map_set for the incremented nonce");
	BOOST_CHECK_MESSAGE(
		useNonceBody.find("\"field\":\"nonces\"") != std::string::npos,
		"_useNonce should update the nonces storage field");
	BOOST_CHECK_MESSAGE(
		useNonceBody.find("\"kind\":\"return\"") != std::string::npos,
		"_useNonce should return the pre-increment nonce value");
}

BOOST_AUTO_TEST_CASE(solcore_export_initializer_modifier_is_lowered_into_function_body)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract C {
					error AlreadyInitialized();

					bool public initialized;
					uint256 public value;

					modifier initializer() {
						if (initialized) revert AlreadyInitialized();
						initialized = true;
						_;
					}

					function initialize(uint256 x) external initializer {
						value = x;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_REQUIRE(solcore["featureFlags"].is_object());
	BOOST_CHECK_EQUAL(solcore["featureFlags"]["modifiers"].get<bool>(), true);
	BOOST_REQUIRE(solcore["functions"].is_array());

	Json const* initializeFn = nullptr;
	for (auto const& fn : solcore["functions"])
	{
		if (fn["name"].get<std::string>() == "initialize")
			initializeFn = &fn;
	}
	BOOST_REQUIRE_MESSAGE(initializeFn != nullptr, "initialize function should exist");
	BOOST_CHECK_MESSAGE(initializeFn->value("has_modifiers", false), "initialize should still record modifier presence");

	std::string initializeBody = initializeFn->dump();
	BOOST_CHECK_MESSAGE(
		initializeBody.find("\"kind\":\"if\"") != std::string::npos,
		"initializer modifier guard should be lowered into initialize");
	BOOST_CHECK_MESSAGE(
		initializeBody.find("\"kind\":\"revert\"") != std::string::npos,
		"initializer modifier revert should be lowered into initialize");
	BOOST_CHECK_MESSAGE(
		initializeBody.find("\"field\":\"initialized\"") != std::string::npos,
		"initializer modifier should write the initialized storage flag");
	BOOST_CHECK_MESSAGE(
		initializeBody.find("\"field\":\"value\"") != std::string::npos,
		"initialize body should still contain the user-authored storage write");
}

BOOST_AUTO_TEST_CASE(solcore_export_address_code_length_lowers_to_extcodesize)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract C {
					function codeLength() external view returns (uint256) {
						return address(this).code.length;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_REQUIRE(solcore["type_decls"].is_array());
	BOOST_REQUIRE(solcore["functions"].is_array());

	bool worldHasCodeSize = false;
	for (auto const& decl : solcore["type_decls"])
	{
		if (decl["name"].get<std::string>() != "WorldState")
			continue;
		std::string declDump = decl.dump();
		worldHasCodeSize =
			declDump.find("\"name\":\"codeSize\"") != std::string::npos &&
			declDump.find("\"kind\":\"mapping\"") != std::string::npos;
	}
	BOOST_CHECK_MESSAGE(
		worldHasCodeSize,
		"WorldState should expose a codeSize mapping for extcodesize lowering");

	Json const* codeLength = nullptr;
	for (auto const& fn : solcore["functions"])
	{
		if (fn["name"].get<std::string>() == "codeLength")
			codeLength = &fn;
	}
	BOOST_REQUIRE_MESSAGE(codeLength != nullptr, "codeLength function should exist");

	std::string body = codeLength->dump();
	BOOST_CHECK_MESSAGE(
		body.find("\"kind\":\"extcodesize\"") != std::string::npos,
		"address(this).code.length should lower to extcodesize");
	BOOST_CHECK_MESSAGE(
		body.find("\"path\":[\"env\",\"thisAddress\"]") != std::string::npos,
		"extcodesize should read the current contract address from CallEnv");
	BOOST_CHECK_MESSAGE(
		body.find("\"field\":\"code\"") == std::string::npos,
		"address.code should not survive as a generic field access");
}

BOOST_AUTO_TEST_CASE(solcore_export_constructor_deployment_semantics)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract C {
					bool public sawZero;

					constructor() {
						sawZero = address(this).code.length == 0;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "C");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_CHECK_EQUAL(solcore["featureFlags"]["constructors"].get<bool>(), true);
	BOOST_REQUIRE_MESSAGE(solcore["constructor"].is_object(), "constructor should be exported");
	BOOST_REQUIRE_MESSAGE(
		solcore["deployedCodeSize"].is_string() || solcore["deployedCodeSize"].is_number_integer(),
		"deployedCodeSize should be exported for constructor deployment semantics");

	std::string ctorDump = solcore["constructor"].dump();
	BOOST_CHECK_MESSAGE(
		ctorDump.find("\"name\":\"constructor\"") != std::string::npos,
		"constructor export should use the constructor name");
	BOOST_CHECK_MESSAGE(
		ctorDump.find("\"kind\":\"extcodesize\"") != std::string::npos,
		"constructor body should lower address(this).code.length to extcodesize");
	BOOST_CHECK_MESSAGE(
		ctorDump.find("\"path\":[\"env\",\"thisAddress\"]") != std::string::npos,
		"constructor extcodesize should read thisAddress from CallEnv");
}

BOOST_AUTO_TEST_CASE(solcore_specific_contract_request_still_compiles)
{
	Json input = Json::object();
	input["language"] = "Solidity";
	input["sources"] = Json::object();
	input["sources"]["fileA"] = Json::object();
	input["sources"]["fileA"]["content"] = R"(
		pragma solidity >=0.8.20;
		contract Caller { uint256 public marker; }
	)";
	input["settings"] = Json::object();
	input["settings"]["outputSelection"] = Json::object();
	input["settings"]["outputSelection"]["fileA"] = Json::object();
	input["settings"]["outputSelection"]["fileA"]["Caller"] = Json::array({"solcore"});

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "Caller");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	BOOST_CHECK_MESSAGE(
		contractResult["solcore"].value("unsupported", false) == false,
		"specific-contract solcore requests should force compilation instead of returning unsupported");
}

BOOST_AUTO_TEST_CASE(solcore_export_known_external_target_metadata_and_foreign_registry)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore", "evm.bytecode.object"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				contract Ownable {
					address internal _owner;

					function owner() public view returns (address) {
						return _owner;
					}
				}

				contract Token {
					mapping(address => uint256) internal _balances;
					mapping(address => mapping(address => uint256)) internal _allowances;

					function transfer(address to, uint256 value) public returns (bool) {
						_balances[msg.sender] -= value;
						_balances[to] += value;
						return true;
					}

					function transferFrom(address from, address to, uint256 value) public returns (bool) {
						uint256 current = _allowances[from][msg.sender];
						if (current != type(uint256).max)
							_allowances[from][msg.sender] = current - value;
						_balances[from] -= value;
						_balances[to] += value;
						return true;
					}
				}

				contract Caller {
					uint256 internal marker;

					function getOwner(Ownable target) external view returns (address) {
						return target.owner();
					}

					function spend(Token token, address to, uint256 amount) external {
						token.transfer(to, amount);
						marker = amount;
					}

					function pull(Token token, address from, address to, uint256 amount) external {
						token.transferFrom(from, to, amount);
						marker = amount;
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "Caller");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	Json const& solcore = contractResult["solcore"];
	BOOST_REQUIRE(solcore["foreign_contracts"].is_array());
	BOOST_REQUIRE(solcore["functions"].is_array());
	BOOST_REQUIRE(solcore["type_decls"].is_array());

	std::string solcoreDump = solcore.dump();
	BOOST_CHECK_MESSAGE(
		solcoreDump.find("\"name\":\"contractStorage\"") != std::string::npos,
		"WorldState should expose contractStorage for known foreign contract lowering");
	// The registry used to also carry a heuristic `kind` classification
	// ("ownable"/"erc20"); that field was removed — consumers now use the
	// per-contract dispatch_entries/methods metadata instead, so only
	// registry inclusion is pinned here.
	BOOST_CHECK_MESSAGE(
		solcoreDump.find("\"id\":\"fileA:Ownable\"") != std::string::npos,
		"foreign contract registry should include Ownable");
	BOOST_CHECK_MESSAGE(
		solcoreDump.find("\"id\":\"fileA:Token\"") != std::string::npos,
		"foreign contract registry should include Token");
	// knownTarget gained function/resolution/resolutionKind/selector fields
	// since these pins were first written; the expectations below are the
	// full current serializations (nlohmann dumps keys alphabetically).
	BOOST_CHECK_MESSAGE(
		solcoreDump.find(
			"\"knownTarget\":{\"contractId\":\"fileA:Ownable\",\"function\":\"owner\",\"mutability\":\"view\","
			"\"resolution\":{\"field\":\"_owner\",\"keyArgOrder\":[],\"kind\":\"storage_getter\",\"returnType\":\"address\",\"slot\":\"0\"},"
			"\"resolutionKind\":\"storage_getter\",\"selector\":\"8da5cb5b\",\"signature\":\"owner()\"}") != std::string::npos,
		"known view calls should carry knownTarget metadata");
	BOOST_CHECK_MESSAGE(
		solcoreDump.find(
			"\"knownTarget\":{\"contractId\":\"fileA:Token\",\"function\":\"transfer\",\"mutability\":\"stateful\","
			"\"resolutionKind\":\"cross_contract\",\"selector\":\"a9059cbb\",\"signature\":\"transfer(address,uint256)\"}") != std::string::npos,
		"known ERC20 transfer calls should carry knownTarget metadata");
	BOOST_CHECK_MESSAGE(
		solcoreDump.find(
			"\"knownTarget\":{\"contractId\":\"fileA:Token\",\"function\":\"transferFrom\",\"mutability\":\"stateful\","
			"\"resolutionKind\":\"cross_contract\",\"selector\":\"23b872dd\",\"signature\":\"transferFrom(address,address,uint256)\"}") != std::string::npos,
		"known ERC20 transferFrom calls should carry knownTarget metadata");
}

BOOST_AUTO_TEST_CASE(solcore_export_try_catch_optioned_contract_call_keeps_known_target)
{
	Json input = generateStandardJson(
		false,
		Json(),
		Json::array({"solcore", "evm.bytecode.object"}),
		SolidityCode({
			{"fileA", R"(
				pragma solidity >=0.8.20;

				interface IBasket {
					function quantity(address erc20) external view returns (uint192);
					function disableBasket() external;
				}

				contract Caller {
					IBasket internal basketHandler;

					function reserveGas() internal pure returns (uint256) {
						return 1000;
					}

					function probe(address erc20) external {
						try basketHandler.quantity{gas: reserveGas()}(erc20) returns (uint192 quantity) {
							if (quantity != 0)
								basketHandler.disableBasket();
						} catch {
							basketHandler.disableBasket();
						}
					}
				}
			)"}
		})
	);

	Json result = compile(input.dump());
	BOOST_REQUIRE(containsAtMostWarnings(result));

	Json contractResult = getContractResult(result, "fileA", "Caller");
	BOOST_REQUIRE(contractResult["solcore"].is_object());
	std::string solcoreDump = contractResult["solcore"].dump();
	BOOST_CHECK_MESSAGE(
		solcoreDump.find("\"function\":\"unknown_call\"") == std::string::npos,
		"try/catch optioned contract calls must not lose callee identity");
	BOOST_CHECK_MESSAGE(
		solcoreDump.find("\"kind\":\"try_catch\"") != std::string::npos,
		"try/catch statement should be exported structurally");
	BOOST_CHECK_MESSAGE(
		solcoreDump.find("\"method\":\"quantity\"") != std::string::npos,
		"try/catch call should preserve the contract method name");
	BOOST_CHECK_MESSAGE(
		solcoreDump.find(
			"\"knownTarget\":{\"contractId\":\"fileA:IBasket\",\"function\":\"quantity\",\"mutability\":\"view\","
			"\"resolutionKind\":\"cross_contract\",\"selector\":\"a5a5828c\",\"signature\":\"quantity(address)\"}") != std::string::npos,
		"optioned try/catch contract calls should carry knownTarget metadata");
	BOOST_CHECK_MESSAGE(
		solcoreDump.find("\"method\":\"disableBasket\"") != std::string::npos,
		"non-optioned statement calls in try/catch branches should still export normally");
}

BOOST_AUTO_TEST_CASE(source_location_of_bare_block)
{
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"A.sol": {
				"content": "contract A { constructor() { uint x = 2; { uint y = 3; } } }"
			}
		},
		"settings": {
			"outputSelection": {
				"A.sol": {
					"A": ["evm.bytecode.sourceMap"]
				}
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));

	solidity::frontend::StandardCompiler compiler;
	Json result = compiler.compile(parsedInput);

	std::string sourceMap = result["contracts"]["A.sol"]["A"]["evm"]["bytecode"]["sourceMap"].get<std::string>();

	// Check that the bare block's source location is referenced.
	std::string sourceRef =
		";" +
		std::to_string(std::string{"contract A { constructor() { uint x = 2; "}.size()) +
		":" +
		std::to_string(std::string{"{ uint y = 3; }"}.size());
	BOOST_REQUIRE(sourceMap.find(sourceRef) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(ethdebug_excluded_from_wildcards)
{
	frontend::StandardCompiler compiler;
	// excluded from output selection wildcard
	Json result = compiler.compile(generateStandardJson(true, {}, Json::array({"*"})));
	BOOST_REQUIRE(result.dump().find("ethdebug") == std::string::npos);
	// excluded from debug info selection wildcard
	result = compiler.compile(generateStandardJson(true, {"*"}, Json::array({"ir"})));
	BOOST_REQUIRE(result.dump().find("ethdebug") == std::string::npos);
	// excluded from both - just in case ;)
	result = compiler.compile(generateStandardJson(true, {"*"}, Json::array({"*"})));
	BOOST_REQUIRE(result.dump().find("ethdebug") == std::string::npos);
}

BOOST_AUTO_TEST_CASE(ethdebug_debug_info_ethdebug)
{
	static std::vector<std::tuple<Json, std::optional<std::function<bool(Json)>>>> tests{
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"*"})),
			std::nullopt,
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebug"}), Json::array({"*"})),
			std::nullopt,
		},
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"evm.bytecode.ethdebug"})),
			std::nullopt,
		},
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"evm.deployedBytecode.ethdebug"})),
			std::nullopt,
		},
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug"})),
			std::nullopt,
		},
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"irOptimized"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"irOptimized"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"irOptimized", "evm.bytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"irOptimized", "evm.deployedBytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"irOptimized", "evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"irOptimized", "evm.bytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"irOptimized", "evm.deployedBytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"irOptimized", "evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebug"}), Json::array({"irOptimized"}), YulCode()),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebug"}), Json::array({"irOptimized"}), YulCode()),
			{}
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebugs"}), Json::array({"irOptimized"}), YulCode()), {}
		},
		{
			generateExperimentalStandardJson(
				true, Json::array({"ethdebug"}), {
					{"fileA", {{"contractA", Json::array({"evm.deployedBytecode.bin"})}}},
					{"fileB", {{"contractB", Json::array({"evm.bytecode.bin"})}}}
				},
				SolidityCode({
					{"fileA", "pragma solidity >=0.0; contract contractA { function f() public pure {} }"},
					{"fileB", "pragma solidity >=0.0; contract contractB { function f() public pure {} }"}
				}), true
			),
			std::nullopt,
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebug"}), Json::array({"*"}), EvmAssemblyCode()),
			std::nullopt,
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebug"}), Json::array({"*"}), SolidityAstCode()),
			std::nullopt,
		},
	};
	frontend::StandardCompiler compiler;
	for (auto const& test: tests)
	{
		Json result = compiler.compile(std::get<0>(test));
		if (std::get<1>(test).has_value())
			BOOST_REQUIRE((*std::get<1>(test))(result));
	}
}

BOOST_AUTO_TEST_CASE(ethdebug_ethdebug_output)
{
	static std::vector<std::tuple<Json, std::optional<std::function<bool(Json)>>>> tests{
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"evm.bytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(false, {}, Json::array({"evm.bytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"evm.deployedBytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(false, {}, Json::array({"evm.deployedBytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(false, Json::array({"ethdebug"}), Json::array({"evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(false, {}, Json::array({"evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(true, Json::array({"location"}), Json::array({"evm.bytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(true, Json::array({"location"}), Json::array({"evm.deployedBytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(true, Json::array({"location"}), Json::array({"evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebug"}), Json::array({"evm.bytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["bytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebug"}), Json::array({"evm.deployedBytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["deployedBytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, Json::array({"ethdebug"}), Json::array({"evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["deployedBytecode"].contains("ethdebug") &&
					 result["contracts"]["fileA"]["C"]["evm"]["bytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.bytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["bytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.deployedBytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["deployedBytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug"})),
			[](const Json& result)
			{
				return result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["deployedBytecode"].contains("ethdebug") &&
					 result["contracts"]["fileA"]["C"]["evm"]["bytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.bytecode.ethdebug", "ir"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos && result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["bytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.deployedBytecode.ethdebug", "ir"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos && result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["deployedBytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.bytecode.ethdebugs"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.deployedBytecode.ethdebugs"})),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug", "ir"})),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos && result.contains("ethdebug") && result["contracts"]["fileA"]["C"]["evm"]["deployedBytecode"].contains("ethdebug") &&
					 result["contracts"]["fileA"]["C"]["evm"]["bytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.bytecode.ethdebug", "ir"}), YulCode()),
			[](const Json& result)
			{
				return result.dump().find("/// ethdebug: enabled") != std::string::npos && result["contracts"]["fileA"]["object"]["evm"]["bytecode"].contains("ethdebug");
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.deployedBytecode.ethdebug", "ir"}), YulCode()),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.bytecode.ethdebug", "evm.deployedBytecode.ethdebug", "ir"}), YulCode()),
			std::nullopt
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.bytecode"})),
			[](const Json& result)
			{
				return result.dump().find("ethdebug") == std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(true, {}, Json::array({"evm.deployedBytecode"})),
			[](const Json& result)
			{
				return result.dump().find("ethdebug") == std::string::npos;
			}
		},
		{
			generateExperimentalStandardJson(
				true, {}, {
					{"fileA", {{"contractA", Json::array({"evm.deployedBytecode.ethdebug"})}}},
					{"fileB", {{"contractB", Json::array({"evm.bytecode.ethdebug"})}}}
				},
				SolidityCode({
					{"fileA", "pragma solidity >=0.0; contract contractA { function f() public pure {} }"},
					{"fileB", "pragma solidity >=0.0; contract contractB { function f() public pure {} }"}
				}), true
			),
			[](const Json& result)
			{
				return result["contracts"]["fileA"]["contractA"]["evm"]["deployedBytecode"].contains("ethdebug") &&
					result["contracts"]["fileB"]["contractB"]["evm"]["bytecode"].contains("ethdebug") && result.contains("ethdebug");
			}
		}
	};
	frontend::StandardCompiler compiler;
	for (auto const& [standardJsonToCompile, optionalCheck]: tests)
	{
		Json result = compiler.compile(standardJsonToCompile);
		BOOST_REQUIRE(!optionalCheck.has_value() ? result.contains("errors") : result.contains("contracts"));
		if (optionalCheck.has_value())
			BOOST_REQUIRE((*optionalCheck)(result));
	}
}

BOOST_DATA_TEST_CASE(ethdebug_output_instructions_smoketest, boost::unit_test::data::make({"deployedBytecode", "bytecode"}), bytecodeType)
{
	frontend::StandardCompiler compiler;
	Json result = compiler.compile(generateExperimentalStandardJson(true, {}, Json::array({std::string("evm.") + bytecodeType + ".ethdebug"})));
	BOOST_REQUIRE(result["contracts"]["fileA"]["C"]["evm"][bytecodeType].contains("ethdebug"));
	bool creation = std::string(bytecodeType) == "bytecode";
	Json ethdebugInstructionsToCheck = result["contracts"]["fileA"]["C"]["evm"][bytecodeType]["ethdebug"];
	BOOST_REQUIRE(ethdebugInstructionsToCheck["contract"]["definition"]["source"]["id"] == 0);
	BOOST_REQUIRE(ethdebugInstructionsToCheck["contract"]["name"] == "C");
	BOOST_REQUIRE(ethdebugInstructionsToCheck["environment"] == (creation ? "create" : "call"));
	BOOST_REQUIRE(ethdebugInstructionsToCheck["instructions"].is_array());
	for (auto const& instruction: ethdebugInstructionsToCheck["instructions"])
	{
		BOOST_REQUIRE(instruction.contains("offset"));
		BOOST_REQUIRE(instruction.contains("operation"));
		BOOST_REQUIRE(instruction["operation"].contains("mnemonic"));
		if (instruction.contains("context"))
		{
			BOOST_REQUIRE(instruction["context"]["code"]["range"].contains("length"));
			BOOST_REQUIRE(instruction["context"]["code"]["range"].contains("offset"));
			BOOST_REQUIRE(instruction["context"]["code"]["source"].contains("id"));
		}
		std::string mnemonic = instruction["operation"]["mnemonic"];
		if (mnemonic.find("PUSH") != std::string::npos)
		{
			size_t bytesToPush = boost::lexical_cast<size_t>(mnemonic.substr(4));
			if (bytesToPush > 0)
			{
				BOOST_REQUIRE(instruction["operation"].contains("arguments"));
				BOOST_REQUIRE(instruction["operation"]["arguments"].is_array());
				BOOST_REQUIRE(instruction["operation"]["arguments"].size() == 1);
				std::string argument = instruction["operation"]["arguments"][0];
				BOOST_REQUIRE(argument.length() % 2 == 0);
				BOOST_REQUIRE(bytesToPush == (argument.length() - 2) / 2); // remove "0x" and calculate actual byte size from hex.
			}
			else
				BOOST_REQUIRE(!instruction["operation"].contains("arguments"));
		}
		else
			BOOST_REQUIRE(!instruction["operation"].contains("arguments"));
	}
}

BOOST_AUTO_TEST_CASE(no_experimental_import_ast_solidity_evmasm)
{
	frontend::StandardCompiler compiler;
	{
		char const* input = R"(
		{
			"language": "SolidityAST",
			"sources": {
				"A": {
		            "assemblyJson": {
		                ".code": [
		                    { "name": "PUSH", "value": "0" }
						]
					}
				}
			}
		}
		)";

		Json parsedInput;
		BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));
		Json result = compiler.compile(parsedInput);
		BOOST_CHECK(containsError(result, "FatalError", "'SolidityAST' and 'EVMAssembly' inputs are experimental and can only be used with the 'settings.experimental' option enabled."));
	}
}

BOOST_AUTO_TEST_CASE(no_experimental_invalid_output_selection)
{
	frontend::StandardCompiler compiler;
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"A.sol": {
				"content": "contract A { constructor() { uint x = 2; { uint y = 3; } } }"
			}
		},
		"settings": {
			"outputSelection": {
				"A.sol": {
					"A": ["irOptimizedAst"]
				}
			}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));
	Json result = compiler.compile(parsedInput);
	BOOST_CHECK(
		containsError(
			result,
			"FatalError", "'irAst', 'irOptimizedAst', 'yulCFGJson', and 'ethdebug' outputs are experimental and can only be used with the 'settings.experimental' option enabled."
		)
	);
}

BOOST_AUTO_TEST_CASE(experimental_non_boolean)
{
	frontend::StandardCompiler compiler;
	char const* input = R"(
	{
		"language": "Solidity",
		"sources": {
			"A.sol": {
				"content": "contract A { constructor() { uint x = 2; { uint y = 3; } } }"
			}
		},
		"settings": {
			"experimental": 1
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));
	Json result = compiler.compile(parsedInput);
	BOOST_CHECK(containsError(result, "JSONError", "'settings.experimental' must be a Boolean."));
}

BOOST_AUTO_TEST_CASE(via_ssa_cfg_with_experimental)
{
	frontend::StandardCompiler compiler;
	static auto constexpr input = R"(
	{
		"language": "Solidity",
		"sources": {
			"A.sol": {
				"content": "// SPDX-License-Identifier: GPL-2.0\npragma solidity >=0.0;\ncontract A { function f() public pure returns (uint) { return 1; } }"
			}
		},
		"settings": {
			"experimental": true,
			"viaSSACFG": true,
			"outputSelection": {"*": {"*": ["evm.bytecode"]}}
		}
	}
	)";

	Json parsedInput;
	BOOST_REQUIRE(util::jsonParseStrict(input, parsedInput));
	Json result = compiler.compile(parsedInput);
	// Should compile without fatal errors (warnings are acceptable)
	BOOST_CHECK(!containsError(result, "FatalError", ""));
}

BOOST_AUTO_TEST_SUITE_END()

} // end namespaces
