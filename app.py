from flask import Flask, render_template, request, jsonify,  send_from_directory, make_response
import subprocess
from reportlab.pdfgen import canvas

from reportlab.pdfgen import canvas
from flask import Response
from reportlab.pdfgen import canvas
from io import BytesIO
from flask import Response
import os
import numpy as np
from tensorflow.keras.models import load_model
import pydicom
import cv2
import numpy as np
import math
from datetime import datetime
import zipfile
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection





app = Flask(__name__)

# Load the saved model
loaded_model = load_model('templates/trained_model_resnet3d.h5')
size = 50
NoSlices = 20
temp_dir =''
# Initialize a global variable to store the progress percentage
progress_percentage = 0
birthdate=""
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global progress_percentage
    if request.method == 'POST':
        # Get uploaded files
        uploaded_files = request.files.getlist("dicom_files")
        # Create a temporary directory to store uploaded files
        temp_dir = 'temp_upload'
        if os.path.exists(temp_dir):
            # Remove all files in the directory
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                os.remove(file_path)
        else:
            # Create the directory if it doesn't exist
            os.makedirs(temp_dir)
        try:
            for filename in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, filename))
            
            print(f"Files in directory '{temp_dir}' have been removed.")
        except OSError as e:
            print(f"Error: {e}")
        #os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files to the temporary directory
        for file in uploaded_files:
            if file.filename != '':
                file.save(os.path.join(temp_dir, file.filename))

        # Process and make predictions on the uploaded DICOM files
        predictions = []
        processedData = []
        cuc = 0
        tuc = len(uploaded_files)
        for filename in os.listdir(temp_dir):
            cuc += 1
            percentage = int((cuc / tuc) * 100)
            update_progress(percentage)  # Update the global progress_percentage

            try:
                processedData.append(dataProcessingUnlabeled(os.path.join(temp_dir), size=size, noslices=NoSlices))
            except Exception as e:
                print('Error processing file:', filename, e)

        x_unlabeled = np.array([processedData])
        x_unlabeled = x_unlabeled.reshape(-1, size, size, NoSlices, 1)
        prediction = loaded_model.predict(x_unlabeled)
        predicted_label = np.argmax(prediction, axis=1)
        predictions.append((filename, predicted_label[0]))
        test = os.path.join(temp_dir, predictions[0][0])
        name = str(pydicom.dcmread(test).PatientID)
        birthdate = str(pydicom.dcmread(test).PatientBirthDate)
        #render("temp_upload")
        # Remove temporary directory and files
        #for filename in os.listdir(temp_dir):
        #    os.remove(os.path.join(temp_dir, filename))
        #os.rmdir(temp_dir)
        progress_percentage = 0

        
        return render_template('result.html', predictions=predictions,name=name,birthdate=birthdate)
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
    return jsonify({"percentage": progress_percentage,
                    "message": "Uploading file...",})

def update_progress(percentage):
    global progress_percentage
    progress_percentage = percentage
    
'''@app.route('/report/<name>/<birthdate>')
def view_report(name, birthdate):
    birthdate =birthdate[:4]+'/'+birthdate[4:6]+'/'+birthdate[-2:]
    return render_template('report.html', name=name, birthdate=birthdate)'''

@app.route('/generate_report/<name>', methods=['GET'])
def generate_report(name):
    # Call the render function here
    temp_dir='temp_upload'
    render(temp_dir)
    # You can also perform any additional processing or rendering here if needed

    return render_template('report.html', name=name, birthdate=birthdate)
  
def load_scan(path):
    slices = []
    for s in os.listdir(path):
        if s.endswith('.dcm'):
            dicom_file = pydicom.dcmread(os.path.join(path, s), force=True)
            if hasattr(dicom_file, 'ImagePositionPatient'):
                slices.append(dicom_file)
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def render(PFpath):
    INPUT_FOLDER = PFpath  # Check if there are any DICOM files
    first_patient = load_scan(os.path.join(INPUT_FOLDER))
    first_patient_pixels = get_pixels_hu(first_patient)
    # Example usage assuming 'first_patient' is a list of 'ScanObject' instances
    first_patient = [ScanObject(SliceThickness=1.0, PixelSpacing=[0.5, 0.5])]
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
    plot_3d(pix_resampled, 400,'temp_images/3d_image1.png')
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    plot_3d(segmented_lungs_fill, 0, 'temp_images/3d_image2.png')
    plot_3d(segmented_lungs_fill - segmented_lungs, 0,'temp_images/3d_image3.png')
class ScanObject:
    def __init__(self, SliceThickness, PixelSpacing):
        self.SliceThickness = SliceThickness
        self.PixelSpacing = PixelSpacing

def resample(image, scan, new_spacing=[1, 1, 1]):
    if not scan:
        raise ValueError("The 'scan' object is empty or not in the expected format.")

    # Check the structure of the 'scan' object and access attributes accordingly
    if isinstance(scan[0], ScanObject):
        slice_thickness = scan[0].SliceThickness
        pixel_spacing = scan[0].PixelSpacing
    else:
        raise AttributeError("The 'scan' object does not have the expected structure.")

    # Determine current pixel spacing
    spacing = np.array([slice_thickness] + pixel_spacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def plot_3d(image, threshold=-300, save_path=None):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)  # Updated line

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
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

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]

    #Fill the air around the person
    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

@app.route('/get_image/<image_name>')
def get_image(image_name):
    image_dir = 'temp_images'
    if image_name in os.listdir(image_dir):
        return send_from_directory(image_dir, image_name)
    else:
        return "Image not found", 404

@app.route('/download_report/<name>')
def generate_pdf(name):
    # Render the HTML content
    html_content = render_template('report_template.html', name=name, birthdate="01/01/1990")

    # Create a temporary HTML file to store the content
    temp_html_file = f"temp_{name}.html"
    with open(temp_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Generate the PDF using wkhtmltopdf
    pdf_file = f"report_{name}.pdf"
    subprocess.run(['wkhtmltopdf', temp_html_file, pdf_file])

    # Clean up the temporary HTML file
    subprocess.run(['rm', temp_html_file])

    # Prepare the PDF response
    with open(pdf_file, 'rb') as f:
        pdf_data = f.read()

    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'inline; filename=report_{name}.pdf'

    # Clean up the PDF file (optional)
    subprocess.run(['rm', pdf_file])

    return response

if __name__ == '__main__':
    app.run(debug=True)
