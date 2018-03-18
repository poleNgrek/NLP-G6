def read_file(path):
	with open(path) as fp:
		yield from read(fp)

def read(fp):
	text = ["<ROOT>"]
	tag = ["<ROOT>"]
	head = [0]
	for line in fp:
		line = line.rstrip() # strip off the trailing newline
		if not line.startswith('#'):
			if len(line) == 0:
				yield (text, tag, head)
				text = ["<ROOT>"]
				tag = ["<ROOT>"]
				head = [0]
			else:
				columns = line.split()
				if columns[0].isdigit(): # skip range tokens
					text += columns[1:2]
					tag += columns[3:4]
					head += [int(columns[6])]


