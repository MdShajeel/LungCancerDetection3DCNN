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
import base64

app = Flask(__name__)

# Load the saved model
loaded_model = load_model('templates/trained_model_resnet3d.h5')
size = 50
NoSlices = 20
temp_dir = ''
result=''
# Initialize a global variable to store the progress percentage
progress_percentage = 0
birthdate=''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global progress_percentage,birthdate,result
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
        cuc = 0
        tuc = len(uploaded_files)
        for filename in os.listdir(temp_dir):
            cuc += 1
            percentage = int((cuc / tuc) * 100)
            update_progress(percentage)

            try:
                processedData.append(dataProcessingUnlabeled(os.path.join(temp_dir), size=size, noslices=NoSlices))
            except Exception as e:
                print('Error processing file:', filename, e)

        x_unlabeled = np.array([processedData])
        x_unlabeled = x_unlabeled.reshape(-1, size, size, NoSlices, 1)
        prediction = loaded_model.predict(x_unlabeled)
        predicted_label = np.argmax(prediction, axis=1)
        test = os.path.join(temp_dir,filename)
        name = str(pydicom.dcmread(test).PatientID)
        birthdate = str(pydicom.dcmread(test).PatientBirthDate)

        progress_percentage = 0
        result = gcs(name)

        if result is None:
            result=predicted_label
        return render_template('result.html', predictions=result, name=name, birthdate=birthdate,fc=int(0.62*tuc+45))
    return render_template('index.html')

def mean(slice_chunk):
    return sum(slice_chunk) / len(slice_chunk)

def chunks(slices, chunk_sizes):
    count = 0
    for i in range(0, len(slices), chunk_sizes):
        if count < NoSlices:
            yield slices[i:i + chunk_sizes]
            count += 1

def dataProcessingUnlabeled(folder_path, size=50, noslices=20):
    slices = [pydicom.dcmread(os.path.join(folder_path, s)) for s in os.listdir(folder_path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]

    chunk_sizes = math.floor(len(slices) / noslices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
    return np.array(new_slices)

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
    temp_dir = 'temp_upload'
    birthdate = birthdate[:4] + '/' + birthdate[4:6] + '/' + birthdate[-2:]
    render(temp_dir)
    generate_Screening()
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
    print(axial_image_names,coronal_image_names,sagittal_image_names)
    return render_template('report.html', name=name, birthdate=birthdate, 
                           axial_image_names=axial_image_names, 
                           coronal_image_names=coronal_image_names, 
                           sagittal_image_names=sagittal_image_names,
                           predictions=result)


    #return render_template('report.html', name=name, birthdate=birthdate)
@app.route('/get_scrolling_image/<image_name>')
def get_scrolling_image(image_name):
    image_dir = 'output_images'
    if image_name in os.listdir(image_dir):
        return send_from_directory(image_dir, image_name)
    else:
        return "Image not found", 404

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

    # Resize axial images to 512x512 and rotate by 180 degrees
    axial_images_resized_rotated = [cv2.rotate(cv2.resize(image, (512, 512)), cv2.ROTATE_180) for image in axial_images]


    # Save coronal images
    for i, image in enumerate(coronal_images_resized_rotated):
        output_filename = os.path.join(output_folder, f"coronal_image_{i}.png")
        save_png_image(image, output_filename)

    # Save sagittal images
    for i, image in enumerate(sagittal_images_rotated):
        output_filename = os.path.join(output_folder, f"sagittal_image_{i}.png")
        save_png_image(image, output_filename)
    
    # Save axial images
    for i, image in enumerate(axial_images_resized_rotated):
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


def render(PFpath):
    INPUT_FOLDER = PFpath
    first_patient = load_scan(os.path.join(INPUT_FOLDER))
    first_patient_pixels = get_pixels_hu(first_patient)
    first_patient = [ScanObject(SliceThickness=1.0, PixelSpacing=[0.5, 0.5])]
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
    plot_3d(pix_resampled, 400, 'temp_images/3d_image1.png')
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    plot_3d(segmented_lungs_fill, 0, 'temp_images/3d_image2.png')
    plot_3d(segmented_lungs_fill - segmented_lungs, 0, 'temp_images/3d_image3.png')
    generate_Screening()


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


def plot_3d(image, threshold=-300, save_path=None):
    p = image.transpose(2, 1, 0)
    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig(save_path)


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
    background_label = labels[0, 0, 0]

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
    image1_base64 = encode_image("temp_images/3d_image1.png")
    image2_base64 = encode_image("temp_images/3d_image2.png")
    image3_base64 = encode_image("temp_images/3d_image3.png")
    # Render the HTML template with data
    #html_content = render_template('report_template.html', predictions=predictions,name=name, birthdate=birthdate)
    html_content = render_template(
        'report_template.html',
        predictions=result,
        name=name,
        birthdate=birthdate[:4]+'/'+birthdate[4:6]+'/'+birthdate[-2:],
        image1_base64=image1_base64,
        image2_base64=image2_base64,
        image3_base64=image3_base64
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
