#! /usr/bin/env python
import logging
log = None
handler = None
LOG_PATH = '../data/log/'
#---------------------------------------
def init(filename):
	global log, handler
	if log and handler : return "Error: log is already initialized. Try log.close() first."
	if type(filename) != str or filename == '' : return "Error: wrong filename"
	log = logging.getLogger();
	log.setLevel(logging.INFO)
	filename = LOG_PATH + str(filename)
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter(
		fmt='%(asctime)s %(levelname)s: %(message)s',
		datefmt='%m-%d %H:%M'
		)
	handler.setFormatter(formatter)
	log.addHandler(handler)
	return
#------------------------------------
def msg(message):
	if not message : return "Error: empty message"
	if not log or not handler : return "Error: log is not initialized"
	log.info(message);
	return
#------------------------------------
def close():
	global log,handler
	if not log or not handler : return "Error: log is already closed."
	log.removeHandler(handler)
	log, handler = None, None
	return
#----------------------------------------
