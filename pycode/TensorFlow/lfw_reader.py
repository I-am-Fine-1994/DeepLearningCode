import os
import os.path
import sys

# folder structure looks like this:
# D:/Database/lfw/
# --lfw-name.txt
# --pairDevTrain.txt
# --pairDevTest.txt
# --image 					# this is a folder
# ----individuals' folders
# ------img_name_0001.jpg
# ------img_name_0002.jpg
# ------img_name_0003.jpg

def notrecommand(func):
	def wrapper(*args, **kw):
		print("This function will return a list type of data, which means more memory will be used, and this may causeing stuck. The XX_gen function is recommended. If you don't understand what a generator is, you can still use this function for convience.")
		return func(*args, **kw)
	return wrapper

class lfw_reader():
	"""
	"""

	def __init__(self):
		self.lfw_path = "D:\Database\lfw"
		self.img_path = "image"
		self.lfw_names_txt = "lfw-names.txt"
		self.pair_train_txt = "pairsDevTrain.txt"
		self.pair_test_txt = "pairsDevTest.txt"

	# this function will return a list containing the lists of image file 
	# under all indivisual's folder
	@notrecommand
	def get_img_file_list(self):
		file_list = []
		for item in self.img_file_list_gen():
			file_list.append(item)
		return file_list
	
	# this function will return a list containing all lists which individual's 
	# name and number of photos
	@notrecommand
	def get_name_list(self):
		names_list = []
		for item in self.name_list_gen():
			names_list.append(item)
		return names_list

	# this function will return a list of pair trainset
	@notrecommand
	def pair_trainset_list(self):
		trainset_list = []
		for item in self.pair_trainset_gen():
			trainset_list.append(item)
		return trainset_list

	# this function will return a list of pair testset
	@notrecommand
	def pair_testset_list(self):
		testset_list = []
		for item in self.pair_testset_gen():
			testset_list.append(item)
		return testset_list

	# this function will return a list containing all images' path
	@notrecommand
	def img_full_path_list(self):
		full_path_list = []
		for path in self.img_full_path_gen():
			full_path_list.append(path)
		return full_path_list

	# this function will return a list containing all images' path of trainset
	@notrecommand
	def pair_trainset_full_path_list():
		full_path_list = []
		for path in self.pair_trainset_full_path_gen():
			full_path_list.append(path)
		return full_path_list
	# this function will return a list containing all images' path of testset
	@notrecommand
	def pair_testset_full_path_list():
		full_path_list = []
		for path in self.pair_testset_full_path_gen():
			full_path_list.append(path)
		return full_path_list

	# this function will return a generator of trainset
	# for the purpose of saving memory
	def pair_trainset_gen(self):
		return self.pair_list_gen(os.path.join(self.lfw_path, self.pair_train_txt))

	# this function will return a generator of testset
	# for the purpose of saving memory
	def pair_testset_gen(self):
		return self.pair_list_gen(os.path.join(self.lfw_path, self.pair_test_txt))

	def img_full_path_gen(self):
		for name in self.name_list_gen():
			name = name[0]
			for img in self.onefolder_list(name):
				# print(img)
				img_full_path = os.path.join(self.lfw_path, self.img_path, name, img)
				# print(img_full_path)
				yield img_full_path

	# this function will return a generator of images' path of trainset
	def pair_trainset_full_path_gen(self):
		return self.pair_full_path_gen(self.pair_train_txt)

	# this function will return a generator of images' path of testset
	def pair_testset_full_path_gen(self):
		return self.pair_full_path_gen(self.pair_test_txt)

	# this function will return a full path of image
	# def img_full_path_gen(self, img_name):
		# img_full_path = os.path.join(self.lfw_path, self.img_path, img_name)
		# return img_full_path

	# this function will return a generator
	# Each item in this generator is:
	# 	filename list under the folder
	# def img_file_list(self):
		# img_folder_path = os.path.join(self.lfw_path, self.img_path)
		# for name in self.name_list_gen():
			# yield os.listdir(os.path.join(img_folder_path, name))

	# this function will return a generator which containing 
	# image pairs' full path
	def pair_full_path_gen(self, txt_name):
		for name1, num1, name2, num2 in self.pair_list_gen(txt_name):
			img1_path = os.path.join(self.lfw_path, self.img_path, name1, name1+"_"+num1+".jpg")
			img2_path = os.path.join(self.lfw_path, self.img_path, name2, name2+"_"+num2+".jpg")
			yield [img1_path, img2_path]

	# return a list which contains the filename under the folder named by
	# parameter "name"
	def onefolder_list(self, name):
		img_folder_path = os.path.join(self.lfw_path, self.img_path)
		onefolder_list = os.listdir(os.path.join(img_folder_path, name))
		return onefolder_list

	# this function will read lfw-names.txt file and return a generator 
	# you can download this txt file from the lfw website
	# what's more, you need to put this file under your lfw_path
	# Each item in this generator is:
	# 	a list consists of individuals' name and the number of photos
	# 	the 1st element is the name, the 2nd element is the number of photos
	def name_list_gen(self):
		filename = os.path.join(self.lfw_path, self.lfw_names_txt)
		with open(filename) as f:
			for name in f:
				yield name.split()

	# this function will parse the txt named pairXX.txt
	# and return a generator of parsed pair_data
	def pair_list_gen(self, pair_txt):
		pair_txt = os.path.join(self.lfw_path, pair_txt)
		with open(pair_txt) as f:
			num = int(f.readline())
			# print(num)
			for line in f:
				pair_data = line.split()
				yield self.parse_pair_data(pair_data)

	# this function will parse the data
	# Return:
	# 	for example: ('Abdullah_Gul_0013', 'Abdullah_Gul_0014')
	def parse_pair_data(self, pair_data):
		if(len(pair_data) == 3):
			name, num1, num2 = pair_data
			name1 = name2 = name
			# return name, num1, num2
		elif(len(pair_data) == 4):
			name1, num1, name2, num2 = pair_data
		return [name1, self.fix_num(num1), name2, self.fix_num(num2)]

	# this function will fix the length of the number
	# such as 12 will be fixed to 0012
	# Return:
	# 	a fixed lenght of number as type str
	def fix_num(self, num, length = 4):
		if type(num) != str:
			num = str(num)
		return num.zfill(length)

# this main function show the most used methods this class lfw_reader
if __name__ == "__main__":
	lr = lfw_reader()
	agen = lr.img_full_path_gen()
	# agen = lr.img_full_path_list)
	for img_path in agen:
		print(img_path)
	