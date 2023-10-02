from orcasong.tools import FileConcatenator
import sys


inputfile =str(sys.argv[1])
outputfile = str(sys.argv[2])

fc = FileConcatenator.from_list(inputfile)

fc.concatenate(outputfile)
