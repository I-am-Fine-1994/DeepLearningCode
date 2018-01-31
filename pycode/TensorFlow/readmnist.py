import struct

def load_train_image():
	print("loading training image set")
	filename = "../mnist/train-images-idx3-ubyte"
	mnist_file = open(filename, 'rb')
	buffers = mnist_file.read()
	
	head = struct.unpack_from('>IIII', buffers, 0);
	offset = struct.calcsize('>IIII')
	img_num = head[1]
	img_height = head[2]
	img_width = head[3]
	
	img_bits = img_num*img_height*img_width
	imgs = struct.unpack_from('>'+str(img_bits)+'B', buffers, offset)
	
	mnist_file.close()
	print("number of images:", img_num)
	print("height of image", img_height)
	print("width of image", img_width)
	return imgs, img_height, img_width, img_num
	
def load_train_label():
	print("loading training label set")
	filename = "../mnist/train-labels-idx1-ubyte"
	
	mnist_file = open(filename, 'rb')
	buffers = mnist_file.read()
	
	head = struct.unpack_from('>II', buffers, 0);
	offset = struct.calcsize('>II')
	label_num = head[1]
	
	label_bits = label_num
	labels = struct.unpack_from('>'+str(label_bits)+'B', buffers, offset)
	
	mnist_file.close()
	
	print("number of labels:", label_num)
	return labels, label_num
	
def load_test_image():
	print("loading test image set")
	filename = "../mnist/t10k-images-idx3-ubyte"
	
	mnist_file = open(filename, 'rb')
	buffers = mnist_file.read()
	
	head = struct.unpack_from('>IIII', buffers, 0);
	offset = struct.calcsize('>IIII')
	img_num = head[1]
	img_height = head[2]
	img_width = head[3]
	
	img_bits = img_num*img_height*img_width
	imgs = struct.unpack_from('>'+str(img_bits)+'B', buffers, offset)
	
	mnist_file.close()
	print("number of images:", img_num)
	print("height of image", img_height)
	print("width of image", img_width)
	return imgs, img_height, img_width, img_num
	
def load_test_label():
	print("loading test label set")
	filename = "../mnist/t10k-labels-idx1-ubyte"
	
	mnist_file = open(filename, 'rb')
	buffers = mnist_file.read()
	
	head = struct.unpack_from('>II', buffers, 0);
	offset = struct.calcsize('>II')
	label_num = head[1]
	
	label_bits = label_num
	labels = struct.unpack_from('>'+str(label_bits)+'B', buffers, offset)
	
	mnist_file.close()
	
	print("number of labels:", label_num)
	return labels, label_num
