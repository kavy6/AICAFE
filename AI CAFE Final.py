import cv2
import numpy as np
import face_recognition
import pyttsx3
import speech_recognition as sr
import textblob

#global variables
name = "Unknown"
num_negatives = 0


#CHATBOT 

#speaktext function for speaking (text-to-speech)
def speaktext(command):
    var_engine = pyttsx3.init()
    print(command)
    var_engine.say(command)
    var_engine.runAndWait()
    
#get audio input from user
def getaudio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('listening...')
        audio = r.listen(source)
        MyText = r.recognize_google(audio)
        print(MyText)
        return MyText        

#conditionals for having a conversation
def speak(reply):
    
    if reply == "hai" or reply == "hi" or reply == "hello":
        speaktext("Hello. How are you feeling today?\n")

    elif reply == "I am good":
        speaktext("Glad to hear that! Would you like to ask me a question?\n")
        
    elif reply == "I feel bad":
        speaktext("Did you take your medicines on time today?\n")
        
    elif reply == "yes I did":
        speaktext("That's good. Did you have a nutritious meal today?\n")
        
    elif reply == "unfortunately I did not":
        speaktext("Oh no!\n")
        speaktext("Connecting you to a caregiver...\n")
        end = True
        return end
    
    elif reply == "yes I ate my meal":
        speaktext("That's great! Would you like to ask me a question?\n")
    
    elif reply == "sadly I did not":
        speaktext("That's bad. You should have your meals on time.\n")
        speaktext("Connecting you to a caregiver...\n")
        end = True
        return end

    elif reply == "what is your name":
        speaktext("I am a bot, would you like to ask me a question?\n")

    elif reply == "what is the full form of AI":
        speaktext("AI stands for Artificial Intelligence\n")

    elif reply == "bye":
        speaktext("Bye and see you!\n")
        end = True
        return end

    else:
        speaktext("How may I help you?\n")
        
#main chatbot function        
def chatbot(name):
    speaktext("Hello, " + name + "\n")
    speaktext("Welcome to AI CAFE!\nHow may I help you?\n")

    end = False
    num_negatives = 0

    while end == False:
        reply = getaudio()
    
        #sentiment analysis
        score = textblob.TextBlob(reply).sentiment.polarity
        
        if score < 0.0:
            num_negatives += 1
            new_end = speak(reply)
            if new_end == True:
                end = True
                
        #human intervention   
        elif num_negatives >= 2:
            speaktext("Connecting you to a caregiver...")
            end = True
            
        else:
            new_end = speak(reply)
            if new_end == True:
                end = True
        


#FACIAL RECOGNITION WITH LIVE VIDEO 


vid = cv2.VideoCapture(0)

#face recognition set up 
face_1 = face_recognition.load_image_file("kavyaphoto.jpeg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]

known_face_encodings = [face_1_encoding]
known_face_names = ["Kavya"]

face_2 = face_recognition.load_image_file("dadphoto.jpeg")
face_2_encoding = face_recognition.face_encodings(face_2)[0]

known_face_encodings.append(face_2_encoding)
known_face_names.append("Dad")


while(True):

    ret, frame = vid.read()

    #face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
     #face recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        for (x,y,w,h) in faces:
            cv2.rectangle(frame , (x,y) , (x+w,y+h) , (0,255,0) , 3)
        
        cv2.putText(frame,name, (left, top+20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1, cv2.LINE_AA)
    
    
    cv2.imshow("image",frame)
    
    

    #authentication
    if name != "Unknown":
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            chatbot(name)            
            break
        
    elif name == "Unknown":
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            speaktext("Authetication error occured")
            break


vid.release()
cv2.destroyAllWindows()

