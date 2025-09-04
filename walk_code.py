from pathlib import Path
import pandas as pd
import ast
import json
import sys
from pprint import pprint

def do_replacement(all_str):
	replacements = {
		'f"': "",
		'"': "",
		"'": "",
		"*args": "\*args",
		"[": "\[",
		"]": "\]",
	}

	for k, v in replacements.items():
		all_str = all_str.replace(k, v)
	
	return all_str

# https://stackoverflow.com/questions/44698193/how-to-get-a-list-of-classes-and-functions-from-a-python-file-without-importing
def get_info(functionNode):
	args = [arg.arg for arg in functionNode.args.args]
	returns = [ast.unparse(n_).replace("return ", "") for n_ in functionNode.body if isinstance(n_, ast.Return)]
	
	return {"args": args, "returns": returns}

def get_str(code_dict, name, indent_level, tab="\t"):
	type = code_dict['type'].upper()

	if code_dict.get("args"):
		args = ', '.join(code_dict["args"])
	else:
		args = ""

	base_indent = tab * (indent_level - 1)
	one_indent = tab * indent_level

	str_rep = f"\n{base_indent}- {type} **{name}** ({args})"

	if code_dict.get("changes"):
		for ch in code_dict["changes"]:
			str_rep += f"\n{one_indent}- {ch}"

	if code_dict.get("returns"):
		returns = ', '.join(code_dict["returns"])
		str_rep += f"\n{one_indent}- returns: {returns}"

	return  str_rep

def format_module_str(all_str, contents):
	for name, code_dict in contents.items():
		all_str += get_str(code_dict, name, indent_level=1)
		all_str += "\n"

		if code_dict['type'] == "class":
			for method_name, method in code_dict['methods'].items():
				all_str += get_str(method, method_name, indent_level=2)
				all_str += "\n"

	return all_str

def get_contents(base_node):
	contents = {}
	for node in base_node.body:
		node_type = "function" if isinstance(node, ast.FunctionDef) else "class" if isinstance(node, ast.ClassDef) else "Unknown"
		
		if node_type == "Unknown":
			continue

		node_name = node.name
		node_dict = {"type": node_type}

		if node_type == "function":
			node_info = get_info(node)
			node_dict.update(node_info)

		elif node_type == "class":
			node_dict["methods"] = {}

			for n_sub in node.body:

				# Walk through methods
				if isinstance(n_sub, ast.FunctionDef):
					method_name = n_sub.name

					# If is init, save these args to overall class args
					if method_name == "__init__":
						node_dict.update(get_info(n_sub))
					# elif method_name == "load":
					# 	breakpoint()

					else:
						node_dict["methods"][n_sub.name] = get_info(n_sub)
						node_dict["methods"][n_sub.name]["type"] = "method"

					node_dict["changes"] = [l.lstrip() for l in ast.unparse(n_sub.body).split("\n") if l.lstrip().startswith("self.") and "=" in l]

		contents[node_name] = node_dict

	return contents

def walk_code(dir):
	python_files = list(Path(str(dir)).glob("**/*.py"))

	all_content = {}
	for file in python_files:
		name = file.stem
		module = file.parts[0]

		if name == "__init__":
			continue

		if ".py" in module:
			module = "None"
				
		# if module not in ["analysis_new", "ecoli_analysis", "_analysis", "analysis", "None"]:
		if module not in ["analysis_new", "ecoli_analysis", "analysis", "None", "make_features"]:
			continue
		
		with open(file) as code:
			try:
				base_node = ast.parse(code.read())
			except Exception as e:
				print("\terror!")

		contents = get_contents(base_node)
		all_content[(module, name)] = contents
	
	return all_content

if __name__ == "__main__":
	write_files = True
	dir = Path(".")

	# -------------------------------------
	#    GET DICTIONARY WITH CODE INFO
	# -------------------------------------
	all_content = walk_code(dir)

	# -------------------------------------
	#      OUTPUT ALL IN ONE MD FILE
	# -------------------------------------
	all_str = ""
	for (module, name), contents in all_content.items():
		all_str += f"\n## {module} - {name}\n"
		all_str = format_module_str(all_str, contents)

	all_str = do_replacement(all_str)
	(dir / "auto_markdown.md").write_text(all_str)

	# -------------------------------------
	#      OUTPUT IN SEPARATE MD FILES
	# -------------------------------------
	file_out_dir = dir / "markdown_files"
	file_out_dir.mkdir(exist_ok=True, parents=True)
	if write_files:
		for (module, script_name), contents in all_content.items():
			
			# if the script has any classes, want to 
			# write each class/method as separate file
			if any([code_dict['type'] == "class" for code_dict in contents.values()]):
				for name, code_dict in contents.items():

					file_name = f"{module} | {script_name} | {name}.md"
					all_str = f"# {name}\n"

					all_str = format_module_str(all_str, {name: code_dict})
					all_str = do_replacement(all_str)
					(file_out_dir / file_name).write_text(all_str)

					

			# otherwise, put everything from script in one file
			else:
				file_name = f"{module} | {script_name}.md"
				all_str = f"# {name}\n"

				all_str = format_module_str(all_str, contents)
				all_str = do_replacement(all_str)
				(file_out_dir / file_name).write_text(all_str)

