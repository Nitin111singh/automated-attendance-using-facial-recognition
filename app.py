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
from flask_mail import Mail, Message
import csv
from flask import flash
from flask import Flask, redirect, url_for, render_template, request, session
import sqlite3
from sklearn import svm
import time

def register_user_to_db(username, password,course):

    cur=mysql.connection.cursor()
    cur.execute("USE users")
    sql="INSERT INTO staff(s_name,pass) values (%s,%s)"
    cur.execute(sql, (username,password))
    mysql.connection.commit()
    sql="INSERT INTO course(s_name,cname) values (%s,%s)"
    cur.execute(sql, (username,course))
    mysql.connection.commit()

    cur.close() 

def check_user(username, password):
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    sql="Select s_name,pass FROM staff WHERE s_name=%s and pass=%s"
    cur.execute(sql, (username, password))

    result = cur.fetchone()
    if result:
       return True
    else:
        return False

#### Defining Flask App
app = Flask(__name__)
app.secret_key = "r@nd0mSk_1"
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'users'
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'
mysql=MySQL(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'nitinsingh.cs20@rvce.edu.in'
app.config['MAIL_PASSWORD'] = 'azsxdcfvgbhn123@'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

#### Saving Date today in 3 different formats
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
    predicted_label = model.predict(facearray.reshape(1, -1))[0]
    trained_labels = model.classes_
    if predicted_label in trained_labels:
        return predicted_label
    else:
        return "unknown"


#### A function which trains the model on all the faces available in faces folder
def train_model():
    # Load face images and labels
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face)
            labels.append(user)
    
    # Convert faces and labels to numpy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    
    # Flatten each face image into a 1D vector
    faces = faces.reshape(len(faces), -1)
    
    # Create an SVM model and train it
    svm_model = svm.SVC(kernel='linear', C=0.01)
    svm_model.fit(faces, labels)
    
    # Save the model to a file
    joblib.dump(svm_model, 'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    value = session.get('username')
     

    file_path = f'Attendance/Attendance-{datetoday}{value}.csv'
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=['Name', 'Roll', 'Time','cname'])
        df.to_csv(file_path, index=False)
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}{value}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    cname=df['cname']
    l = len(df)
    return names,rolls,times,cname,l


#### Add Attendance of a specific user
def add_attendance(name):
    #database query to fetch the course name from given logged in session's user name
    value=session.get('username')
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    sql="select cname from course where s_name=(%s)"
    cur.execute(sql,(value,))
    result = cur.fetchall()
    result = result[0][0]
    print(result)
    cur.close()
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}{value}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}{value}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time},{result}')


################## ROUTING FUNCTIONS #########################

#### Our main page

@app.route('/')
def index():  
    return render_template('login.html')

@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        course =request.form['course']

        register_user_to_db(username, password,course)
        return redirect(url_for('index'))

    else:
        return render_template('register.html')


@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print(check_user(username, password))
        if check_user(username, password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return redirect(url_for('login'))
    else:
        return redirect(url_for('index'))
    
@app.route('/home', methods=['POST', "GET"])
def home():
    if 'username' in session:
        names,rolls,times,cname,l = extract_attendance()
        if session['username']=="admin":
           return  render_template('admin.html',names=names,rolls=rolls,times=times,l=l,cname=cname,totalreg=totalreg(),username=session['username'],datetoday2=datetoday2) 
        else:
             return render_template('home1.html',names=names,rolls=rolls,times=times,l=l,cname=cname,totalreg=totalreg(),username=session['username'],datetoday2=datetoday2) 

    else:
        return "Username or Password is wrong!"
    
@app.route('/logout', methods=["POST", "GET"])
def logout():
    session.clear()
    return redirect(url_for('index'))
 
#### This function will run when we click on Take Attendance Button
import time

@app.route('/start',methods=['GET'])
def start():
    if 'username' in session:
        if 'face_recognition_model.pkl' not in os.listdir('static'):
            return render_template('home1.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.')
   
        cap = cv2.VideoCapture(0)
        ret = True
        attendance_dict = {}
        while ret:
            ret, frame = cap.read()
            if extract_faces(frame) != ():
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1,-1))
                if identified_person is not None:
                    if identified_person not in attendance_dict:
                        attendance_dict[identified_person] = time.time()
                    else:
                        if time.time() - attendance_dict[identified_person] >= 5:
                            add_attendance(identified_person)
                            cv2.putText(frame, f'{identified_person}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                            attendance_dict[identified_person] = time.time()
                        else:
                            cv2.putText(frame, f'{identified_person}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                else:
                    attendance_dict.clear()
                    cv2.putText(frame, 'Unknown', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
           
            cv2.imshow('Attendance',frame)
            if cv2.waitKey(1)==27:
                break
        cap.release()
        cv2.destroyAllWindows()
        names, rolls, times, cname, l = extract_attendance()   
        return redirect(url_for('home'))
    else:
        return "login first"



@app.route('/addd',methods=['GET','POST'])
def addd():
    train_model()
    return "data added"
#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newuseremail=request.form['newuseremail']
    
    #database
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    sql_check="SELECT * FROM user WHERE id=%s"
    cur.execute(sql_check, (newuserid,))
    result=cur.fetchone()
    if result:
        flash("User already exists with this ID")
        return redirect(url_for('home'))
    else:
        sql_insert="INSERT INTO user values (%s,%s,%s)"
        cur.execute(sql_insert, (newusername, newuserid,newuseremail))
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
            cv2.putText(frame,f'Images Captured: {i}/200',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==2000:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,cname,l = extract_attendance()    
    return render_template('home1.html',names=names,rolls=rolls,times=times,cname=cname,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

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
    cur.execute("select * from user where id not in (select id from attendance where day=%s)", (datesql,))
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
        msg.body="you are absent today"
        mail.send(msg)
        cur.close() 
    return "email sent success"
    return render_template('viewattendance.html')

@app.route('/viewattendance',methods=['GET','POST'])

def viewatt():
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    cur.execute("select * from user")
    records=cur.fetchall()

    print("\nPrinting each row")
    for row in records:
        print("name=", row[0], )
        print("id=", row[1])
        print("email=", row[2],"\n")
    
    #database query to fetch the course name from given logged in session's user name
    value=session.get('username')
    cur=mysql.connection.cursor()
    cur.execute("USE users")
    sql="select cname from course where s_name=(%s)"
    cur.execute(sql,(value,))
    result = cur.fetchall()
    result = result[0][0]

    cur=mysql.connection.cursor()
    cur.execute("USE users")
    cur.execute("select * from user where id not in (select id from attendance where day=%s and cname=%s)", (datesql,result))
    records=cur.fetchall()
    cur.execute("select * from user where id in (select id from attendance where day=%s and cname=%s)", (datesql,result))
    presentees=cur.fetchall()


    return render_template('viewattendance.html', records=records,presentees=presentees)

@app.route('/saveattendance',methods=['GET','POST'])
def saveatt():
     #database
    # Open the csv  file
    value=session.get('username')
    print(value)
    with open(f'Attendance/Attendance-{datetoday}{value}.csv', 'r') as file:
        reader = csv.reader(file)
        
        try:
                next(reader)
                next(reader)
        except StopIteration:
                # Handle case where there are no rows in the file
            "<script>alert('No data found in the file');</script>"
            return redirect(url_for('home'))
        for row in reader:
        # If the id is found, insert the id, date, and 1 as the present value into the attendance table
           cur = mysql.connection.cursor()
           cur.execute("USE users")
           sql = "INSERT INTO attendance values(%s,%s,%s,%s)"
           cur.execute(sql, (row[1], datesql, 1,row[3]))
           mysql.connection.commit()
           cur.close()
    return "<script>alert('Data added successfully');</script>"
    

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)