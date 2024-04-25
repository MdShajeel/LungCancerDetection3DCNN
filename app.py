from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
import subprocess
import os
import numpy as np
from tensorflow.keras.models import load_model
from fpi import gcs
import pydicom
import cv2
import numpy as np
import math
import numpy as np
import pydicom
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import base64

app = Flask(__name__)

# Load the saved model
loaded_model = load_model('templates/3d_image_classification.h5')
size = 128
NoSlices = 64
temp_dir = ''
result=''
conversion_factor=0
nodule_area=0
axial_image_names=[]
# Initialize a global variable to store the progress percentage
progress_percentage = 0
birthdate=''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global progress_percentage,birthdate,result,processedData,conversion_factor,nodule_area
    if request.method == 'POST':
        # Get uploaded files
        uploaded_files = request.files.getlist("dicom_files")
        # Create a temporary directory to store uploaded files
        temp_dir = 'temp_upload'
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                os.remove(file_path)
        else:
            os.makedirs(temp_dir)

        try:
            for filename in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, filename))
            for filename in os.listdir("temp_images"):
                os.remove(os.path.join("temp_images", filename))
            print(f"Files in directory '{temp_dir}' have been removed.")
        except OSError as e:
            print(f"Error: {e}")

        # Save uploaded files to the temporary directory
        for file in uploaded_files:
            if file.filename != '':
                file.save(os.path.join(temp_dir, file.filename))

        # Process and make predictions on the uploaded DICOM files
        processedData = []
        test=''
        filename= os.listdir(temp_dir)[0]

        processData=(dataProcessingUnlabeled((temp_dir), size=size, noslices=NoSlices))
        processedData.append([processData])
        x_unlabeled = np.array([data[0]for data in processedData])
        x_unlabeled = x_unlabeled.reshape(-1, size, size, NoSlices, 1)
        prediction = loaded_model.predict(x_unlabeled)
        predicted_label = np.argmax(prediction, axis=1)
        predicted_label = "Cancer" if predicted_label == 1 else "No Cancer"
        test = os.path.join(temp_dir,filename)
        dicom_data = pydicom.dcmread(test)
        name = str(pydicom.dcmread(test).PatientID)
        conversion_factor=dicom_data.PixelSpacing[0]*dicom_data.PixelSpacing[1]
        birthdate = str(pydicom.dcmread(test).PatientBirthDate)

        progress_percentage = 0
        result = gcs(name)

        if result is None:
            result=predicted_label
        return render_template('result.html', predictions=result, name=name, birthdate=birthdate,fc=25)
    return render_template('index.html')

def mean(slice_chunk):
    return sum(slice_chunk) / len(slice_chunk)

def chunks(slices, chunk_sizes):
    count = 0
    for i in range(0, len(slices), chunk_sizes):
        if count < NoSlices:
            yield slices[i:i + chunk_sizes]
            count += 1
# Define window levels and widths for CT scans (adjust these)
window_level = 500  # Adjust this based on your specific requirements
window_width = 1500  # Adjust this based on your specific requirements

def dataProcessingUnlabeled(folder_path, size=128, noslices=64, white_threshold=0.3):
    window_level = 500  # Adjust this based on your specific requirements
    window_width = 1500  # Adjust this based on your specific requirements
    slices = [pydicom.dcmread(os.path.join(folder_path, s)) for s in os.listdir(folder_path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    new_slices = []
    if len(slices) < noslices+30:
        slices.extend([slices[-1]] * (noslices - len(slices)))
        
    uploaded_files = request.files.getlist("dicom_files")
    # Start processing from the specified start_slice
    slices = slices[30:noslices+30]
    cuc = 0
    tuc = len(uploaded_files)
    for slice in slices:
        cuc += 1
        percentage = int((cuc / tuc) * 100)
        update_progress(percentage)
        # Extract pixel array and apply windowing
        pixel_array = np.array(slice.pixel_array)
        windowed_image = np.clip(pixel_array, window_level - window_width/2, window_level + window_width/2)

        # Normalize pixel values to the range [0, 1]
        windowed_image = (windowed_image - (window_level - 0.5 * window_width)) / window_width

        # Extract ROI within lung area and draw contours
        lung_roi, lung_area_with_contours = extract_lung_roi_with_contours(windowed_image)
        # Extract ROI within lung area and draw contours
        lung_mask = extract_lung_roi(lung_roi)
        
        
        # Convert original image to BGR format
        image_bgr = cv2.cvtColor(lung_roi, cv2.COLOR_GRAY2BGR)

        # Apply bitwise AND operation to preserve lung area
        lung_roi = cv2.bitwise_and(image_bgr, image_bgr, mask=lung_mask)

        # If lung mask contains only black regions, subtract it from the original image
        if np.all(lung_mask == 0):
            lung_roi = cv2.subtract(image_bgr, lung_roi)
         # Convert to grayscale
        gray_image = cv2.cvtColor(lung_roi, cv2.COLOR_BGR2GRAY)
        # Normalize pixel values to the range [0, 1]
        normalized_image = gray_image / 255.0
        resized_image = cv2.resize(normalized_image, (size, size))
        
        
       # cv2.imwrite(r"F:\Downloads\test.png",lung_roi)
        # Resize the lung ROI to the desired size (128x128)
        #resized_image = cv2.resize(lung_roi, (size, size))
        
        white_pixels = np.sum(lung_roi >= white_threshold)
        total_pixels = lung_roi.size
        proportion_white = white_pixels / total_pixels
        
        if not (proportion_white >= 0.2 and proportion_white <=0.4):
            #print("skip above")
            continue  # Skip this slice if majority of pixels are white
        else:
#             print("White pixels:", white_pixels)
#             print("Total pixels:", total_pixels)
#             print("Proportion of white pixels:", proportion_white)
            new_slices.append(resized_image)

            
    if len(new_slices) < noslices:
        padding_slices = [new_slices[-1] if new_slices else np.zeros((size, size, 3))] * (noslices - len(new_slices))
        new_slices.extend(padding_slices)


    
    return np.array(new_slices)

# Function to extract ROI within lung area and draw contours
def extract_lung_roi_with_contours(image):
    # Convert image to uint8 data type
    image = (image * 255).astype(np.uint8)
    
    # Apply thresholding to segment lung area
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform morphological operations to enhance lung area
    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of lung area
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and shape
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Filter contours based on size and circularity
        if area > min_area_threshold and circularity > min_circularity_threshold:
            filtered_contours.append(contour)
    
    # Draw contours on original image
    lung_area_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(lung_area_with_contours, filtered_contours, -1, (0, 255, 0), 2)
    
    # Create a mask for the largest contour (assumed to be the lung area)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    
    # Apply mask to original image to extract ROI
    roi = cv2.bitwise_and(image, mask)
    
    return roi, lung_area_with_contours

# Define minimum area threshold for contour filtering
min_area_threshold = 1000  # Adjust this threshold based on your CT images
# Define minimum circularity threshold for contour filtering
min_circularity_threshold = 0.5  # Adjust this threshold based on your CT images

# Function to extract lung ROI by segmenting lung area and removing trachea
def extract_lung_roi(image):
    # Convert image to uint8 data type
    image = (image * 255).astype(np.uint8)
    
    # Apply thresholding to segment lung area
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform morphological operations to enhance lung area
    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of lung area
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to remove small noisy regions
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
    
    # Create a mask for the lung contours
    lung_mask = np.zeros_like(image)
    cv2.drawContours(lung_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    
    # Invert the mask so that the lung area is white and the rest is black
    #lung_mask = cv2.bitwise_not(lung_mask)
    
    return lung_mask

@app.route('/progress', methods=['GET'])
def get_progress():
    global progress_percentage
    return jsonify({"percentage": progress_percentage})


def update_progress(percentage):
    global progress_percentage
    progress_percentage = percentage
for filename in os.listdir("output_images"):
    os.remove(os.path.join("output_images", filename))

@app.route('/generate_report/<name>/<birthdate>', methods=['GET'])
def generate_report(name, birthdate):
    global axial_image_names
    temp_dir = 'temp_upload'
    birthdate = birthdate[:4] + '/' + birthdate[4:6] + '/' + birthdate[-2:]
    plot_html1,plot_html2,plot_html3=render(temp_dir)
    generate_Screening()
    nodule_area=0
    if result=='Cancer':
        nodule_area=mask_cell() 
        nodule_area=int(nodule_area*conversion_factor) if nodule_area!=0 else "NA"
    # Get a list of image names in the output_images folder
    image_folder = 'output_images'
    image_names = os.listdir(image_folder)
    
    # Separate image names based on their view types
    axial_image_names = [img for img in image_names if img.startswith('axial_image')]
    coronal_image_names = [img for img in image_names if img.startswith('coronal_image')]
    sagittal_image_names = [img for img in image_names if img.startswith('sagittal_image')]
    
    # Sort image names
    axial_image_names.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    coronal_image_names.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    sagittal_image_names.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    #print(axial_image_names,coronal_image_names,sagittal_image_names)
    if os.path.exists("temp_images\masked_image.png"):
        status=True
    else:
        status=False
    return render_template('report.html', name=name, birthdate=birthdate, 
                           axial_image_names=axial_image_names, 
                           coronal_image_names=coronal_image_names, 
                           sagittal_image_names=sagittal_image_names,
                           predictions=result,plot_html1=plot_html1,plot_html2=plot_html2,plot_html3=plot_html3,masked_image_exists=status,size=nodule_area)


    #return render_template('report.html', name=name, birthdate=birthdate)
@app.route('/get_scrolling_image/<image_name>')
def get_scrolling_image(image_name):
    image_dir = 'output_images'
    if image_name in os.listdir(image_dir):
        return send_from_directory(image_dir, image_name)
    else:
        return "Image not found", 404


def sort_slices_by_image_position(slices):
    return sorted(slices, key=lambda x: int(x.ImagePositionPatient[2]))
# def generate_Screening():
#     path = 'temp_upload/'
#     output_video = 'temp_images/output_video.mp4' # Output video file name

#     # Get a list of DICOM files in the directory and sort them using the helper function
#     dicom_files = [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.dcm')]
#     dicom_slices = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
#     sorted_slices = sort_slices_by_image_position(dicom_slices)

#     # Read the first DICOM image to get dimensions
#     first_dicom_img = sorted_slices[0]
#     width, height = first_dicom_img.Columns, first_dicom_img.Rows

#     # Create a VideoWriter object to write the images into a video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the video
#     fps = 5  # Adjust the frame rate (fps) to control the video speed
#     video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

#     for dicom_img in sorted_slices:
#         dicom_pixels = dicom_img.pixel_array

#         dicom_pixels = cv2.normalize(dicom_pixels, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

#         video_writer.write(cv2.cvtColor(dicom_pixels, cv2.COLOR_GRAY2BGR))
#     video_writer.release()

# Preprocessing function
def preprocess_image(image):
    # Apply windowing to enhance contrast (e.g., lung window)
    # Adjust these values based on the characteristics of your CT images
    window_center = 500
    window_width = 1500  # Adjusted window width
    # Extract pixel data and convert to NumPy array
    image = image.pixel_array
    # Clip pixel values within the windowing range
    image = np.clip(image, window_center - window_width / 2, window_center + window_width / 2)
    
    # Normalize pixel values to the range [0, 255] for visualization
    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    
    return image

# Function to save PNG image
def save_png_image(image, filename):
    cv2.imwrite(filename, image)

def save_slices_as_images(slices, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sorted_slices = sort_slices_by_image_position(slices)  # Sort the slices

    # Preprocess each image for the current patient
    preprocessed_images = [preprocess_image(slice) for slice in sorted_slices]

    # Reorient images to coronal and sagittal views
    coronal_images = np.transpose(preprocessed_images, (1, 0, 2))  # Transpose from (slice, width, height) to (width, slice, height)
    sagittal_images = np.transpose(preprocessed_images, (2, 0, 1))  # Transpose from (slice, width, height) to (height, slice, width)

    # Exclude specified slices from coronal images
    coronal_images = coronal_images[100:401, :, :]

    # Resize coronal images to 512x512 and rotate by 180 degrees
    coronal_images_resized_rotated = [cv2.rotate(cv2.resize(image, (512, 512)), cv2.ROTATE_180) for image in coronal_images]

    # Exclude specified slices from sagittal images
    sagittal_images = sagittal_images[30:486, :, :]

    # Resize sagittal images to 512x512 and rotate by 90 degrees
    sagittal_images_rotated = [cv2.flip(cv2.transpose(cv2.resize(image, (512, 512))), 1) for image in sagittal_images]

    # Exclude specified slices from axial images
    axial_images = preprocessed_images  # Adjust slice range as needed

    # Resize axial images to 512x512 without rotation
    axial_images_resized = [cv2.resize(image, (512, 512)) for image in axial_images]



    # Save coronal images
    for i, image in enumerate(coronal_images_resized_rotated):
        output_filename = os.path.join(output_folder, f"coronal_image_{i}.png")
        save_png_image(image, output_filename)

    # Save sagittal images
    for i, image in enumerate(sagittal_images_rotated):
        output_filename = os.path.join(output_folder, f"sagittal_image_{i}.png")
        save_png_image(image, output_filename)
    
    # Save axial images
    for i, image in enumerate(axial_images_resized):
        output_filename = os.path.join(output_folder, f"axial_image_{i}.png")
        save_png_image(image, output_filename)





    # for i, dicom_img in enumerate(sorted_slices):
    #     dicom_pixels = dicom_img.pixel_array
    #     dicom_pixels = cv2.normalize(dicom_pixels, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    #     image_filename = os.path.join(output_folder, f"image_{i}.png")
    #     cv2.imwrite(image_filename, dicom_pixels)

# Usage:
# Replace 'temp_upload/' with your DICOM file path
def generate_Screening():
    path = 'temp_upload/'
    output_folder = 'output_images/'  # Change to the folder where you want to save the images
    dicom_files = [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.dcm')]
    dicom_slices = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
    save_slices_as_images(dicom_slices, output_folder)


def load_scan(path):
    slices = []
    for s in os.listdir(path):
        if s.endswith('.dcm'):
            dicom_file = pydicom.dcmread(os.path.join(path, s), force=True)
            if hasattr(dicom_file, 'ImagePositionPatient'):
                slices.append(dicom_file)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)



class ScanObject:
    def __init__(self, SliceThickness, PixelSpacing):
        self.SliceThickness = SliceThickness
        self.PixelSpacing = PixelSpacing

def resample(image, scan, new_spacing=[1, 1, 1]):
    if not scan:
        raise ValueError("The 'scan' object is empty or not in the expected format.")

    if isinstance(scan[0], ScanObject):
        slice_thickness = scan[0].SliceThickness
        pixel_spacing = scan[0].PixelSpacing
    else:
        raise AttributeError("The 'scan' object does not have the expected structure.")

    spacing = np.array([slice_thickness] + pixel_spacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def resample(image, scan, new_spacing=[1, 1, 1]):
    if not scan:
        raise ValueError("The 'scan' object is empty or not in the expected format.")

    if isinstance(scan[0], ScanObject):
        slice_thickness = scan[0].SliceThickness
        pixel_spacing = scan[0].PixelSpacing
    else:
        raise AttributeError("The 'scan' object does not have the expected structure.")

    spacing = np.array([slice_thickness] + pixel_spacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

# first_patient = [ScanObject(SliceThickness=1.0, PixelSpacing=[0.5, 0.5])]
# pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)
    background_label = labels[0,0,0]
    binary_image[background_label == labels] = 2

    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None:
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1
    binary_image = 1 - binary_image

    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image


def plot_3d(image, threshold=-300):
    p = image.transpose(2, 1, 0)
    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = go.Figure(data=[go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                     color='lightpink', opacity=0.5)])
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1), 
                                 camera_eye=dict(x=1.2, y=1.2, z=1.2)))
    return fig.to_html(full_html=False)


def render(PFpath):
    INPUT_FOLDER = PFpath
    first_patient = load_scan(os.path.join(INPUT_FOLDER))
    first_patient_pixels = get_pixels_hu(first_patient)
    first_patient = [ScanObject(SliceThickness=1.0, PixelSpacing=[0.5, 0.5])]
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
    
    img3d1=plot_3d(pix_resampled, 400)
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    img3d2=plot_3d(segmented_lungs_fill, 0)
    img3d3=plot_3d(segmented_lungs_fill - segmented_lungs, 0)
    return img3d1,img3d2,img3d3

def mask_cell():
    # Assuming you have processed image data in 'imageData'
    largest_area = 0
    largest_area_slice = None
    largest_area_contour = None

    # Iterate through each processed image slice
    for idx, processed_image in enumerate(processedData[0][0]):
        # Convert the image to grayscale
        gray_image = (processed_image * 255).astype(np.uint8)
        
        # Apply thresholding to segment the lung area
        _, lung_mask = cv2.threshold(gray_image, 200, 250, cv2.THRESH_BINARY)
        
        # Invert the lung mask
        lung_mask = cv2.bitwise_not(lung_mask)
        
        # Apply the lung mask to the processed image
        lung_image = cv2.bitwise_and(gray_image, gray_image, mask=lung_mask)
        
        # Apply thresholding again to segment the nodules within the lung area
        _, nodule_mask = cv2.threshold(lung_image, 200, 250, cv2.THRESH_BINARY)
        
        # Find contours of the nodules within the lung area
        nodule_contours, _ = cv2.findContours(nodule_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on aspect ratio and area
        filtered_contours = []
        for contour in nodule_contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            
            # Filter contours based on aspect ratio and area threshold
            if aspect_ratio > 0.5 and area > 100 and area < 700:
                filtered_contours.append(contour)
        
        # Find the contour with the largest area for this slice
        if filtered_contours:
            largest_contour = max(filtered_contours, key=cv2.contourArea)
            nodule_area = cv2.contourArea(largest_contour)
            
            # Check if this slice has the largest area so far
            if nodule_area > largest_area and nodule_area < 700:
                largest_area = nodule_area
                largest_area_slice = idx
                largest_area_contour = largest_contour

                print("Contour Area in slice", idx, ":", nodule_area)
                
    # Display the lung image with the largest contour area less than 1000
    if largest_area_contour is not None:
        largest_area_slice_image = processedData[0][0][largest_area_slice]
        largest_area_slice_image_uint8 = (largest_area_slice_image * 255).astype(np.uint8)
        
        # Apply thresholding again for better visualization
        _, largest_area_slice_mask = cv2.threshold(largest_area_slice_image_uint8, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours of the largest contour in the lung area
        largest_area_slice_contours, _ = cv2.findContours(largest_area_slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out only the largest contour within the area range (100 to 1000)
        filtered_largest_contours = []
        for contour in largest_area_slice_contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 1000:
                filtered_largest_contours.append(contour)
        
        # Draw the filtered largest contour on the lung image
        largest_area_slice_contour_image = cv2.cvtColor(largest_area_slice_image_uint8, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(largest_area_slice_contour_image, filtered_largest_contours, -1, (0, 255, 0), 2)
        
        # Display the lung image with the filtered largest contour area and its contour
        plt.imshow(largest_area_slice_contour_image)
        plt.title("Cancer cell Area Contour")
        plt.axis('off')

        # Save the figure before displaying it
        plt.savefig('temp_images/masked_image.png')

        # Display the original lung image without any contour
        plt.imshow(largest_area_slice_image, cmap='gray')  # Assuming largest_area_slice_image is the original lung image
        plt.title("Original Lung Image")
        plt.axis('off')

        # Save the figure before displaying it
        plt.savefig('temp_images/org_image.png')
        return nodule_area
    else:
        print("No cancer found.")
    return 0


@app.route('/get_image/<image_name>')
def get_image(image_name):
    image_dir = 'temp_images'
    if image_name in os.listdir(image_dir):
        return send_from_directory(image_dir, image_name)
    else:
        return "Image not found", 404

# @app.route('/get_video')
# def get_video():
#     video_path = 'temp_images/output_video.mp4'  # Adjust the path to your video file
#     return send_from_directory('', video_path)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


@app.route('/download_report/<name>')
def generate_pdf(name):
    image1_base64=''
    image2_base64=''
    if os.path.exists("temp_images\masked_image.png"):
        status=True
        image1_base64 = encode_image("temp_images/masked_image.png")
        image2_base64 = encode_image("temp_images/org_image.png")
    else:
        status=False
    axial_image_base64_list = []
    for image_name in axial_image_names:
        image_path = os.path.join('output_images', image_name)
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            axial_image_base64_list.append(image_base64)
    # Render the HTML template with data
    #html_content = render_template('report_template.html', predictions=predictions,name=name, birthdate=birthdate)
    html_content = render_template(
        'report_template.html',
        predictions=result,
        name=name,
        birthdate=birthdate[:4]+'/'+birthdate[4:6]+'/'+birthdate[-2:],
        masked_image_exists=status,
        image1_base64=image1_base64,
        image2_base64=image2_base64,
        size=nodule_area,
        axial_image_base64_list=axial_image_base64_list
    )
    # Define file paths
    temp_html_file = f"templates/temp_{name}.html"
    pdf_file = f"templates/report_{name}.pdf"

    # Write the HTML content to a temporary HTML file
    with open(temp_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    try:
        subprocess.run(['C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe', temp_html_file, pdf_file])
    except Exception as e:
        return str(e), 500  # Return an error message if the conversion fails

    # Remove the temporary HTML file
    os.remove(temp_html_file)

    # Serve the PDF as a download
    with open(pdf_file, 'rb') as f:
        pdf_data = f.read()

    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'inline; filename=report_{name}.pdf'

    # Remove the temporary PDF file
    os.remove(pdf_file)

    return response
@app.route('/gerneartepage/<name>')
def generate_page(name):
    image1_base64=''
    image2_base64=''
    if os.path.exists("temp_images\masked_image.png"):
        status=True
        image1_base64 = encode_image("temp_images/masked_image.png")
        image2_base64 = encode_image("temp_images/org_image.png")
    else:
        status=False
    axial_image_base64_list = []
    for image_name in axial_image_names:
        image_path = os.path.join('output_images', image_name)
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            axial_image_base64_list.append(image_base64)
    return render_template('report_template.html',
        predictions=result,
        name=name,
        birthdate=birthdate[:4]+'/'+birthdate[4:6]+'/'+birthdate[-2:],
        masked_image_exists=status,
        image1_base64=image1_base64,
        image2_base64=image2_base64,
        size=nodule_area,
        axial_image_base64_list=axial_image_base64_list)
'''API_KEY = 'f5e0c74f008d47f8b07c76d7c962397'

@app.route('/generate_pdf/<name>')
def generate_pdf(name):
    # Render the HTML template with data
    html_content = render_template('report.html', name=name, birthdate=birthdate)

    # Define PDFShift API endpoint
    pdfshift_url = 'https://api.pdfshift.io/v3/convert/pdf/'

    # Create the request headers with your API key
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    # Create the request payload
    payload = {
        'source': html_content,
        'landscape': False,  # Adjust orientation as needed
        'use_print': False  # Set to True if you want to use print styles
    }

    # Send a POST request to PDFShift to convert HTML to PDF
    response = requests.post(pdfshift_url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Create a response with PDF content
        pdf_data = response.content
        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'inline; filename=report_{name}.pdf'
        return response
    else:
        return 'Error converting HTML to PDF', 500'''

if __name__ == '__main__':
    app.run(debug=True)
