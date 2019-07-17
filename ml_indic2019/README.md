<b>Training using google ml engine-</b>

import os

PROJECT = 'indic2019' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'indic2019' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1

os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION

%%bash
gcloud config set project $PROJECT
gcloud config set compute/region $REGION

%%bash
gcloud ml-engine jobs submit training job_anand1 \
    --runtime-version=1.13 \
    --python-version=3.5 \
    --module-name=train_anand.task \
    --package-path=train_anand/ \
    --staging-bucket=gs://$BUCKET \
    --region=$REGION \
    -- \
    --train-steps=100 \
    --eval-steps=3 \
    --verbosity DEBUG
   
   <b>Write on cloud firestore using datalab</b>
  #!pip install firebase_admin
    import firebase_admin
    from firebase_admin import credentials
    from firebase_admin import firestore

    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
      'projectId': "indic2019",
    })

    db = firestore.client()
    cred = credentials.Certificate('gs://indic2019-authorize/indic2019-ba5d5d1a3723.json')
    firebase_admin.initialize_app(cred)

    data = {
    u'name': u'Los Angeles',
    u'state': u'CA',
    u'country': u'USA'
    }

    # Add a new doc in collection 'cities' with ID 'LA'
    db.collection(u'cities').document(u'LA').set(data)
    print('data') 
    
 <b>Copy file from Datalab repository to Bucket</b>
 !gsutil cp -r 'export_dir' gs://indic2019
 
Note- Few dependencies have to be removed while training the model via Google-ml-cloud. It supports Estimator for training and post training the model can be used from Google AI Serving.
