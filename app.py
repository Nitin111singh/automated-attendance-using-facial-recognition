import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_mysqldb import MySQL
import yaml
from flask_mail import Mail, Message
import openpyxl
import csv

#### Defining Flask App
app = Flask(__name__)
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'users'
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'
mysql=MySQL(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'nitinsingh.cs20@rvce.edu.in'
app.config['MAIL_PASSWORD'] = 'Nitin@8899865679'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
datesql=date.today().strftime("%y-%m-%d")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
           (x,y,w,h) = extract_faces(frame)[0]
           cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
           face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
           identified_person = identify_face(face.reshape(1,-1))[0]
           if identified_person is not None:
              add_attendance(identified_person)
              cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)

           else:
              cv2.putText(frame,f'Unknown',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)

        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

@app.route('/addd',methods=['GET','POST'])
def addd():
    train_model()
#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newuseremail=request.form['newuseremail']
    
    #database
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    sql="INSERT INTO user values(%s,%s,%s)"
    cur.execute(sql, (newusername, newuserid,newuseremail))
    mysql.connection.commit()
    cur.close() 

    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

@app.route('/database',methods=['GET','POST'])
def database():
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    cur.execute("INSERT INTO user values('nitin',1,'nitinsingh.cs20@rvce.edu.in')")
    mysql.connection.commit()
    cur.close() 
    return 'successfully entered data'

@app.route('/sendmail',methods=['GET','POST'])
def sendmail():
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    cur.execute("select * from user")
    records=cur.fetchall()
    # for row in records:
    #     print(row)
    
    print("\nPrinting each row")
    for row in records:
        print("name=", row[0], )
        print("id=", row[1])
        print("email=", row[2],"\n")
       
    for row in records:
        msg = Message("hii",sender='nitinsingh.cs20@rvce.edu.in',recipients=[row[2]])
        msg.body="hi!! you're absent today"
        mail.send(msg)
        cur.close() 
    return 'email sent'

@app.route('/viewattendance',methods=['GET','POST'])

def viewatt():
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    cur.execute("select * from user")
    records=cur.fetchall()
                      #database
    # Open the Excel file
    with open(f'Attendance/Attendance-{datetoday}.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip the header row
        for row in reader:
        # If the id is found, insert the id, date, and 1 as the present value into the attendance table
           cur = mysql.connection.cursor()
           cur.execute("USE users")
           sql = "INSERT INTO attendance values(%s,%s,%s)"
           cur.execute(sql, (row[1], datesql, 1))
           mysql.connection.commit()
           cur.close()

    print("\nPrinting each row")
    for row in records:
        print("name=", row[0], )
        print("id=", row[1])
        print("email=", row[2],"\n")


    return render_template('viewattendance.html', records=records)


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)