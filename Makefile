##
# Project Title
#
# @file
# @version 0.1
makeFileDir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

init:
	pip3 install -r requirements.txt
clean:
	rm -r $(makeFileDir)__pycache__
# end
