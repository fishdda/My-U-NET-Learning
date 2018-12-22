import os
import cv2
import glob
from tqdm import tqdm
import pydicom as dicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dicompylercore import dicomparser, dvh, dvhcalc
from skimage.draw import polygon
from skimage.transform import resize
import sys
import random



class DCM_TO_MASK:

	def __init__(self,dicom_path,train_path,test_path):

		self.path = dicom_path
		self.train = train_path
		self.test = test_path


	def Read_Dicom(self,file_list):

		dcm_file = os.listdir(os.path.join(self.path,file_list))
		self.DCM,self.RT,self.DT = [],[],[]

		for item in dcm_file:
		    
		    if item.split('.')[-1] == 'dcm':
		        
		        dcm = dicom.read_file(os.path.join(self.path,file_list,item),force=True)
		        
		        if dcm.Modality == 'CT':

		            self.DCM.append(dcm)

		        elif dcm.Modality == 'RTSTRUCT':

		            self.RT.append(dcm)

		        else:

		            self.DT.append(dcm)
		        
		self.DCM.sort(key = lambda x: float(x.ImagePositionPatient[2]))

		for s in self.DCM: s.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
		
		self.image = np.stack([s.pixel_array for s in self.DCM])

	    # Convert to int16
		self.image = self.image.astype(np.int16)

	    # Set outside-of-scan pixels to 0
		self.image[self.image == -2000] = 0


	    # Convert to Hounsfield units(HU)
		for slice_number in range(len(self.DCM)):

		    intercept = self.DCM[slice_number].RescaleIntercept
		    slope = self.DCM[slice_number].RescaleSlope

		    if slope != 1:

		        self.image[slice_number] = slope * self.image[slice_number].astype(np.float64)
		        self.image[slice_number] = self.image[slice_number].astype(np.int16)

		    self.image[slice_number] += np.int16(intercept)


		return self.DCM,self.image,self.RT

	def read_structure(self,structure):
	    
	    self.contours = []
	    for i in range(len(structure[0].ROIContourSequence)):
	        
	        contour = {}
	        contour['color'] = structure[0].ROIContourSequence[i].ROIDisplayColor
	        contour['number'] = structure[0].ROIContourSequence[i].ReferencedROINumber
	        contour['name'] = structure[0].StructureSetROISequence[i].ROIName
	        assert contour['number'] == structure[0].StructureSetROISequence[i].ROINumber
	        
	        contour['contours'] = [s.ContourData for s in structure[0].ROIContourSequence[i].ContourSequence]
	        self.contours.append(contour)
	        
	    return self.contours


	def get_mask(self,dicom,image,contours):
	    '''
	       You can get mask about parotid
	    '''
	    Z = [s.ImagePositionPatient[2] for s in dicom] #所有Z方向上的坐标
	    pos_r = dicom[0].ImagePositionPatient[1]
	    spacing_r = dicom[0].PixelSpacing[1]
	    pos_c = dicom[0].ImagePositionPatient[0]
	    spacing_c = dicom[0].PixelSpacing[0]
	    
	    self.label = np.zeros_like(image,dtype = np.int16)
	    for con in contours:

	        if con['name'] == 'Bladder':
	            num = int(con['number'])

	            for c in con['contours']:
	                nodes = np.array(c).reshape((-1,3))
	                assert np.amax(np.abs(np.diff(nodes[:,2]))) == 0
	                z_index = Z.index(nodes[0,2])
	                r = (nodes[:,1] - pos_r) / spacing_r
	                c = (nodes[:,0] - pos_c) / spacing_c
	                rr,cc = polygon(r,c)
	                self.label[z_index,rr,cc] = num # 
	        else:
	            pass
	    self.colors = tuple(np.array([con['color'] for con in contours]) / 255.0)



	    return self.label,self.colors

	def Generate_Train(self,IMG_DIM):

		## To generate the training data
		IMG_W = 128
		IMG_H = 128
		IMG_CHANNELS = 3

		self.img = np.zeros((1,IMG_DIM[0],IMG_DIM[1]))
		self.mask = np.zeros((1,IMG_DIM[0],IMG_DIM[1]))
		file = os.listdir(self.path)
		print(file)
		for item in file:
			if item != '.DS_Store':
				for nam in os.listdir(os.path.join(self.path,item)):

					if nam.split(".")[-1] == 'npy'and nam.split("_")[0] == 'images':

						img = np.load(os.path.join(self.path,item,nam))
						print (img.shape)
						self.img = np.concatenate((self.img,img),axis = 0)

					if nam.split(".")[-1] == 'npy'and nam.split("_")[0] == 'masks':

						mask = np.load(os.path.join(self.path,item,nam))
						self.mask = np.concatenate((self.mask,mask),axis = 0)

		self.img = self.img[1:,:,:]
		self.mask = self.mask[1:,:,:]

		self.X_train = np.ones((self.img.shape[0],IMG_W,IMG_H,IMG_CHANNELS),dtype=np.uint8)
		self.Y_train = np.ones((self.img.shape[0],IMG_W,IMG_H,1),dtype=np.bool)

		for i in range(self.img.shape[0]):
			
			## note: one channel image should transfer to dtype = np.uint8
			img_ = np.array(cv2.resize(self.img[i],(IMG_W,IMG_H),interpolation = cv2.INTER_CUBIC), dtype=np.uint8)

			self.X_train[i] = cv2.cvtColor(img_,cv2.COLOR_GRAY2BGR)
			self.Y_train[i] = cv2.resize(self.mask[i],(IMG_W,IMG_H),interpolation = cv2.INTER_CUBIC).reshape(IMG_W,IMG_H,1)


		np.save(os.path.join(self.train,"images_train.npy"), self.X_train)
		np.save(os.path.join(self.train,"masks_train.npy"), self.Y_train)
		print ("Train Images Dimension:{}".format(self.X_train.shape))
		print ("Train Masks Dimension:{}".format(self.Y_train.shape))

	# Define IoU metric
	def mean_iou(self,y_true, y_pred):

	    prec = []
	    for t in np.arange(0.5, 1.0, 0.05):
	        y_pred_ = tf.to_int32(y_pred > t)
	        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
	        K.get_session().run(tf.local_variables_initializer())
	        with tf.control_dependencies([up_opt]):
	            score = tf.identity(score)
	        prec.append(score)

	    return K.mean(K.stack(prec), axis=0)

	def Model(self):
		# Build U-Net model
		inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
		s = Lambda(lambda x: x / 255) (inputs)

		self.c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
		self.c1 = Dropout(0.1) (c1)
		self.c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c1)
		self.p1 = MaxPooling2D((2, 2)) (self.c1)

		self.c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.p1)
		self.c2 = Dropout(0.1) (self.c2)
		self.c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c2)
		self.p2 = MaxPooling2D((2, 2)) (self.c2)

		self.c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.p2)
		self.c3 = Dropout(0.2) (self.c3)
		self.c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c3)
		self.p3 = MaxPooling2D((2, 2)) (self.c3)

		self.c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.p3)
		self.c4 = Dropout(0.2) (self.c4)
		self.c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c4)
		self.p4 = MaxPooling2D(pool_size=(2, 2)) (self.c4)

		self.c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.p4)
		self.c5 = Dropout(0.3) (self.c5)
		self.c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c5)

		self.u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (self.c5)
		self.u6 = concatenate([self.u6, self.c4])
		self.c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.u6)
		self.c6 = Dropout(0.2) (self.c6)
		self.c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c6)

		self.u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (self.c6)
		self.u7 = concatenate([self.u7, self.c3])
		self.c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.u7)
		self.c7 = Dropout(0.2) (self.c7)
		self.c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c7)

		self.u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (self.c7)
		self.u8 = concatenate([self.u8, self.c2])
		self.c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.u8)
		self.c8 = Dropout(0.1) (c8)
		self.c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c8)

		self.u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (self.c8)
		self.u9 = concatenate([self.u9, self.c1], axis=3)
		self.c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.u9)
		self.c9 = Dropout(0.1) (self.c9)
		self.c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (self.c9)

		outputs = Conv2D(1, (1, 1), activation='sigmoid') (self.c9)

		self.model = Model(inputs=[inputs], outputs=[outputs])
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
		self.model.summary()



	def test_img(self):
		# Check if training data looks all right

		X_train = np.load(os.path.join(self.train,"images_train.npy"))
		Y_train = np.load(os.path.join(self.train,"masks_train.npy"))

		ix_ = random.sample(range(0,X_train.shape[0]+1),3)
		ix = random.randint(0, X_train.shape[0])
		plt.figure(figsize=(10,5))
		plt.subplot(231)
		plt.imshow(X_train[ix_[0]],'gray')
		plt.subplot(234)
		plt.imshow(Y_train[ix_[0],:,:,0],'gray')
		plt.subplot(232)
		plt.imshow(X_train[ix_[1]],'gray')
		plt.subplot(235)
		plt.imshow(Y_train[ix_[1],:,:,0],'gray')
		plt.subplot(233)
		plt.imshow(X_train[ix_[2]],'gray')
		plt.subplot(236)
		plt.imshow(Y_train[ix_[2],:,:,0],'gray')
		plt.show()





def main():

	## Image Preprocessing
	dicom_path = '/Users/henryhuang/Desktop/deep learning/10prostate/10prostate/'
	test_path = '/Users/henryhuang/Desktop/deep learning/10prostate/py/test_data'
	train_path = '/Users/henryhuang/Desktop/deep learning/10prostate/py/train_data'

	IMG_WIDTH = 512
	IMG_HEIGHT = 512
	IMG_DIM = (IMG_WIDTH,IMG_HEIGHT)


	X = DCM_TO_MASK(dicom_path,train_path,test_path)
	# list_ = os.listdir(dicom_path)
	# print(list_)
	# print('Getting and resizing train images and masks ... ')
	# for i,item in enumerate(list_):
	# 	if item != '.DS_Store':
			
	# 		DCM,image,RT = X.Read_Dicom(item)

	# 		print("{0}th Image Shape:{1}".format(i+1,image.shape))
	# 		img_inf = [DCM[0].Rows,DCM[0].Columns,int(DCM[0].SliceThickness)]
	# 		print('CT image Rows*Columns*SliceThickness: {}'.format(img_inf))
	# 		id_ = int(DCM[0].PatientID)
			

	# 		contours = X.read_structure(RT)
	# 		label,colors = X.get_mask(DCM,image,contours)

	# 		## Save the image data to original path
	# 		np.save(os.path.join(dicom_path,str(id_),"images_%d.npy" % (id_)), image)
	# 		np.save(os.path.join(dicom_path,str(id_),"masks_%d.npy" % (id_)), label)

	# X.Generate_Train(IMG_DIM)
	# print('Done!')
	X.test_img()



if __name__ == '__main__':
	main()