import pandas as pd
import re
#Collate the data contents and clear or escape the xml string
def preprocessLine(inputLine):
	"""
	:param inputLine:
	:return: clumn(clean data)
	"""
	a_link = '<[^>]*>'
	clumn = ''
	body = inputLine
	#Clear the xml tag content or escape content from the text
	if type(body) == float:
		clumn = ''
	else:
		clumn = re.sub('\n',' ',body)
		while re.search('&amp;',clumn):
			clumn = clumn.replace('&amp;', '&')
		clumn = clumn.replace('&gt;', '>')
		clumn = clumn.replace('&lt;', '<')
		clumn = clumn.replace('&apos;', ',')
		clumn = clumn.replace('&quot;', '"')
		clumn = clumn.replace('&nbsp;',' ')
		clumn = clumn.replace('&ndash;','-')
		clumn = clumn.replace('&euro;', '€')
		clumn = clumn.replace('&mdash;', '—')
		clumn = clumn.replace('&thinsp;','')
		clumn = clumn.replace('&pm;','±')
		clumn = clumn.replace('&#xA;',' ')
		clumn = re.sub(a_link, ' ', clumn)
		clumn = re.sub(" +", " ", clumn)
	return clumn
#This function is responsible for reading the data, calling the handler, and writing the cleaned data
def splitFile(inputFile, outputFile_question, outputFile_answer):
	"""
	:param inputFile:
	:param outputFile_question:
	:param outputFile_answer:
	:return:Output answer.txt and question.txt
	"""
	answer = []
	question = []
	else_list = []
	clumn = ''
	#Read the data and store it to data
	try:
		data = pd.read_xml(inputFile)
	except  IOError:
		print("This is a IOError")
		return False
	id = data.Id
	type_id = data.PostTypeId
	body = data.Body
	question_path = outputFile_question
	answer_path = outputFile_answer
	#Pass the data to the cleaning data function, and filter answer question and others article types and store them
	for i in range(len(id)):
		clumn = preprocessLine(body[i])
		if type_id[i] == 1:
			question.append(clumn)
		elif type_id[i] == 2:
			answer.append(clumn)
		else:
			else_list.append(clumn)
	#Create and write the article content separately
	try:
		answer_value = open(answer_path,'w',encoding='utf-8')
		question_value = open(question_path,'w',encoding='utf-8')
		for i in answer:
			answer_value.write(i + '\n')
		for i in question:
			question_value.write(i + '\n')
		answer_value.close()
		question_value.close()
	except IOError:
		print("This is a IOError")
		return False
#Start program main
if __name__ == "__main__":
	# Pass parameters and content
	f_data = "data.xml"
	f_question = "question.txt"
	f_answer = "answer.txt"
	splitFile(f_data, f_question, f_answer)