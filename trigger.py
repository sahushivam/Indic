#the function was deployed as cloud function which triggers and get the coordinates of bounding box and store it in firestore.
import os
import firebase_admin
import googleapiclient.discovery
from firebase_admin import credentials
from google.cloud import firestore
from firebase_admin import firestore

def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, vergoogleapiclientsion of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

def hello_gcs(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    
    project="indic2019"
    model="yolo_2"
    version="yolo_2_4"
    instances=[{"image_bytes": "gs://indic2019/10_10.jpg" }]
    
    file = event
    print(f"Processing file: {file['name']}.")
    filename=str(file['name'])
    print(filename)
    f=predict_json(project,model,instances,version)
    co=str(f[0]['coordinates'])
    db = firestore.Client()
    doc_ref = db.collection(u'file').document(filename)
    doc_ref.set({
        u'filename': filename,
        u'coordinate': co,
    })






# Function dependencies, for example:
# package>=version

# google-cloud-core
# google-cloud-firestore
# firebase-admin
# oauth2client
# google-api-python-client


