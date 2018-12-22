import os
import cv2
import glob
import pydicom as dicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dicompylercore import dicomparser, dvh, dvhcalc
from skimage.draw import polygon




class DCM_TO_MASK:

	def __init__(self,dicom_path,train_path,test_path,train_data):

		self.path = dicom_path
		self.train = train_path
		self.test = test_path
		self.train_data = train_data


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

	def Generate_Train(self):

		self.img = np.ones((1,512,512))
		self.mask = np.ones((1,512,512))
		path = os.listdir(self.train)
		for item in path: 
			if item.split('_')[0] == 'images':

				img = np.load(os.path.join(self.train,item))
				self.img = np.concatenate((self.img,img),axis = 0)

			elif item.split('_')[0] == 'masks':

				mask = np.load(os.path.join(self.train,item))
				self.mask = np.concatenate((self.mask,mask),axis = 0)


		np.save(os.path.join(self.train_data,"images_train.npy"), self.img[1:,:,:])
		np.save(os.path.join(self.train_data,"masks_train.npy"), self.mask[1:,:,:])
		print ("Train Images Dimension:{}".format(self.img[1:,:,:].shape))
		print ("Train Masks Dimension:{}".format(self.mask[1:,:,:].shape))




def main():
	dicom_path = '/Users/henryhuang/Desktop/deep learning/10prostate/10prostate/'
	train_path = '/Users/henryhuang/Desktop/deep learning/10prostate/py/train'
	test_path = '/Users/henryhuang/Desktop/deep learning/10prostate/py/test'
	train_data = '/Users/henryhuang/Desktop/deep learning/10prostate/py/train_data'
	X = DCM_TO_MASK(dicom_path,train_path,test_path,train_data)
	# list_ = os.listdir(dicom_path)
	# print(list_)
	# for i,item in enumerate(list_):
	# 	if item != '.DS_Store':
			
	# 		DCM,image,RT = X.Read_Dicom(item)

	# 		print("{0}th Image Shape:{1}".format(i+1,image.shape))
	# 		img_inf = [DCM[0].Rows,DCM[0].Columns,int(DCM[0].SliceThickness)]
	# 		print('CT image Rows*Columns*SliceThickness: {}'.format(img_inf))
	# 		id_ = int(DCM[0].PatientID)
	# 		np.save(os.path.join(train_path,"images_%d.npy" % (id_)), image)

	# 		contours = X.read_structure(RT)
	# 		label,colors = X.get_mask(DCM,image,contours)

	# 		np.save(os.path.join(train_path,"masks_%d.npy" % (id_)), label)

	X.Generate_Train()

	print ('Done')


if __name__ == '__main__':
	main()