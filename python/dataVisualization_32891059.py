import re
import matplotlib.pyplot as plt
from parser_32891059 import  Parser
#This function is responsible for reading data and collecting processed data for visualization to produce pictures
#Pictures show the vocabulary of each passage
def visualizeWordDistribution(inputFile, outputImage):
	"""
	:param inputFile:
	:param outputImage:
	:return:
		Return to picture
	"""
	line = []
	vocabulary_size = []
	visualize = []
	x_list = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100','others']
	other = 0
	#Read the file and collect the data
	try:
		data_value = open(inputFile, 'r',encoding='utf-8')
		for i in data_value:
			line.append(i)
		data_value.close()
	except IOError:
		print("This is a IOError")
		return False
	#Pass the collected data to the handler and get the word count
	for i in line:
		line_clean = Parser(i)
		if line_clean.getVocabularySize() != None:
			vocabulary_size.append(line_clean.getVocabularySize()//10)
	#Count the number of numbers in each segment
	for i in range(10):
		visualize.append(vocabulary_size.count(i))
	#Count the number of articles that are 100 or greater
	for j in visualize:
		other = other + j
	#Visual display and storage
	visualize.append(len(vocabulary_size)-other)
	plt.bar(x_list, visualize)
	plt.title("Vocabulary distribution of posts")
	plt.xlabel("Number of words")
	plt.ylabel("Article number")
	plt.xticks(rotation=30)
	plt.tight_layout()
	plt.savefig(outputImage)
	plt.show()
#This function is responsible for reading data and collecting processed data for visualization to produce pictures
#The figure is a line chart showing the number of answers and questions for each quarter
def visualizePostNumberTrend(inputFile, outputImage):
	"""
	:param inputFile:
	:param outputImage:
	:return:
		Return to picture
	"""
	line = []
	q_a = [[],[]]
	list = [[],[]]
	num_a = 0
	num_b = 0
	#Read the file and collect the data
	try:
		data_value = open(inputFile, 'r', encoding='utf-8')
		for i in data_value:
			if re.search('<row', i):
				line.append(i)
			else:
				continue
		data_value.close()
	except IOError:
		print("This is a IOError")
		return False
	# Pass the collected data to the handler and get the date
	for i in line:
		string = Parser(i)

		if str(string).split('PostType:')[1].split(' creation date:')[0] != '':
			list[0].append(str(string).split('PostType:')[1].split(' creation date:')[0])
			list[1].append(str(string).split('date:')[1].split(' the main')[0])
	#Store the number of responses and questions for each quarter
	for i in range(len(list[1])):
		if list[0][i] == 'Answer':
			num_a +=1
		if list[0][i] == 'Question':
			num_b +=1
		if i == len(list[1]) - 1:
			q_a[0].append(num_a)
			q_a[1].append(num_b)
		if i+1 < len(list[1]):
			if  list[1][i] != list[1][i+1]:
				q_a[0].append(num_a)
				q_a[1].append(num_b)
				num_a = 0
				num_b = 0
	# Visual display and storage
	plt.plot(sorted(set(list[1])),q_a[0], color='orangered', marker='o', linestyle='-', label='Answer_number')
	plt.plot(sorted(set(list[1])),q_a[1] , color='green', marker='D', linestyle='-', label='Question_number')
	plt.xlabel("Time quarter")
	plt.ylabel("Number")
	plt.title('The trend of the number of posts over time')
	plt.legend()
	plt.xticks(rotation=30)
	plt.tight_layout()
	plt.savefig(outputImage)
	plt.show()
#Start program main
if __name__ == "__main__":
	#Pass parameters and content
	f_data = "data.xml"
	f_wordDistribution = "wordNumberDistribution.png"
	f_postTrend = "postNumberTrend.png"
	visualizeWordDistribution(f_data, f_wordDistribution)
	visualizePostNumberTrend(f_data, f_postTrend)