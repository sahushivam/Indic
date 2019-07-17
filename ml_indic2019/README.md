<b>Training using google ml engine-</b>

import os

PROJECT = 'indic2019' # REPLACE WITH YOUR PROJECT ID<br>
BUCKET = 'indic2019' # REPLACE WITH YOUR BUCKET NAME<br>
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1<br>

os.environ['PROJECT'] = PROJECT<br>
os.environ['BUCKET'] = BUCKET<br>
os.environ['REGION'] = REGION<br>

%%bash
gcloud config set project $PROJECT<br>
gcloud config set compute/region $REGION<br>

%%bash
gcloud ml-engine jobs submit training job_anand1 \<br>
    --runtime-version=1.13 \<br>
    --python-version=3.5 \<br>
    --module-name=train_anand.task \<br>
    --package-path=train_anand/ \<br>
    --staging-bucket=gs://$BUCKET \<br>
    --region=$REGION \<br>
    -- \<br>
    --train-steps=100 \<br>
    --eval-steps=3 \<br>
    --verbosity DEBUG<br>
   
   <b>Write on cloud firestore using datalab</b><br>
  #!pip install firebase_admin<br>
    import firebase_admin<br>
    from firebase_admin import credentials<br>
    from firebase_admin import firestore<br>

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
