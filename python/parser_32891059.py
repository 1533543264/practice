import re
#Responsible for cleaning data
class Parser:
	"""docstring for ClassName"""
	def __init__(self, inputString):
		"""
		:param inputString
		"""
		self.inputString = inputString
		self.ID = self.getID()
		self.type = self.getPostType()
		self.dateQuarter = self.getDateQuarter()
		self.cleanBody = self.getCleanedBody()
	def __str__(self):
		#print ID, Question/Answer/Others, creation date, the main content
		return "ID:%s  PostType:%s creation date:%s the main content%s" %(self.ID,self.type,self.dateQuarter,self.cleanBody)
	#Get the id
	def getID(self):
		"""
		:return: line_string (value: id)
		"""
		if re.search('<row',self.inputString):
			line_string = self.inputString.split('Id="')[1].split('" P')[0]
		else:
			return
		return line_string
	#Filtering Article Types
	def getPostType(self):
		"""
		:return: text_type
		"""
		text_type = ''
		if re.search('<row', self.inputString):
			line_string = self.inputString.split('PostTypeId="')[1].split('" C')[0]
		else:
			return
		if line_string == '1':
			text_type ='Question'
		elif line_string == '2':
			text_type = 'Answer'
		else:
			text_type = 'Others'
		return text_type
	#Obtained date quarter
	def getDateQuarter(self):
		"""
		:return: date_string
		"""
		if re.search('<row', self.inputString):
			line_string = self.inputString.split('-',2)
		else:
			return
		char_month = ''
		if line_string[1][0] == '0':
			if int(line_string[1][1]) == 1 or int(line_string[1][1]) == 2 or int(line_string[1][1]) == 3:
				char_month = 'Q1'
			elif int(line_string[1][1]) == 4 or int(line_string[1][1]) == 5 or int(line_string[1][1]) == 6:
				char_month = 'Q2'
			elif int(line_string[1][1]) == 7 or int(line_string[1][1]) == 8 or int(line_string[1][1]) == 9:
				char_month = 'Q3'
		else:
			char_month = 'Q4'
		date_string = line_string[0].split('CreationDate="')[-1]+char_month
		return date_string
	#Data cleaning
	def getCleanedBody(self):
		"""
		:return: line_string
		"""
		a_link = '<[^>]*>'
		if re.search('<row', self.inputString):
			line_string = self.inputString
		else:
			return
		line_string = line_string.split('Body="')[-1]
		line_string = line_string.split('" />')[0]
		while re.search('&amp;',line_string):
			line_string = line_string.replace('&amp;', '&')
		line_string = line_string.replace('&gt;', '>')
		line_string = line_string.replace('&lt;', '<')
		line_string = line_string.replace('&apos;', ',')
		line_string = line_string.replace('&quot;', '"')
		line_string = line_string.replace('&nbsp;',' ')
		line_string = line_string.replace('&ndash;','-')
		line_string = line_string.replace('&euro;', '€')
		line_string = line_string.replace('&mdash;', '—')
		line_string = line_string.replace('&thinsp;','')
		line_string = line_string.replace('&pm;','±')
		line_string = line_string.replace('&#xA;',' ')
		line_string = re.sub(a_link, ' ', line_string)
		line_string = re.sub(" +", " ", line_string)
		return line_string
	#Wash the data and get the number of characters
	def getVocabularySize(self):
		"""
		:return: count_char
		"""
		a_link = '<(.*?)>'
		str_clean = '[^\u0041-\u005a\u0061-\u007a\u0030-\u0039\' ]+'
		if re.search('<row', self.inputString):
			line_string = self.inputString
		else:
			return
		line_string = line_string.split('Body="')[-1]
		line_string = line_string.split('" />')[0]
		while re.search('&amp;',line_string):
			line_string = line_string.replace('&amp;', '&')
		line_string = line_string.replace('&gt;', '>')
		line_string = line_string.replace('&lt;', '<')
		line_string = line_string.replace('&apos;', ',')
		line_string = line_string.replace('&quot;', '"')
		line_string = line_string.replace('&nbsp;',' ')
		line_string = line_string.replace('&ndash;','-')
		line_string = line_string.replace('&euro;', '€')
		line_string = line_string.replace('&mdash;', '—')
		line_string = line_string.replace('&thinsp;','')
		line_string = line_string.replace('&pm;','±')
		line_string = line_string.replace('&#xA;',' ')
		line_string = re.sub(a_link, ' ', line_string)
		line_string = re.sub(" +", " ", line_string)
		line_string = re.sub(str_clean,'',line_string)
		line_string = re.sub('\'', ' ', line_string)
		line_string = re.sub(" +", " ", line_string).strip()
		count_char = line_string.strip().count(' ')+1
		return count_char