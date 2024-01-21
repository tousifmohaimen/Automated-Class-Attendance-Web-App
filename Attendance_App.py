import streamlit as st
import redis
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from insightface.app import FaceAnalysis
import av
import re
import base64
from streamlit_webrtc import webrtc_streamer
from utils import RealTimePred, RegistrationForm

# Connect to Redis
redis_hostname = ''
redis_portnumber = 
redis_password = ''
redis_client = redis.StrictRedis(
    host=redis_hostname,
    port=redis_portnumber,
    password=redis_password
)

# redis-cli -h redis-14895.c74.us-east-1-4.ec2.cloud.redislabs.com -p 14895 -a GNAkOLmU26w6MuyLBfqfAWJcHAXdba4r


def retrive_data(name):
    try:
        retrive_dict = redis_client.hgetall(name)
        retrive_series = pd.Series(retrive_dict)
        retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
        index = retrive_series.index
        index = list(map(lambda x: x.decode(), index))
        retrive_series.index = index
        retrive_df = retrive_series.to_frame().reset_index()
        retrive_df.columns = ['name_id', 'facial_features']

        # Create separate columns for 'Name' and 'Matric'
        retrive_df['Name'] = retrive_df['name_id'].apply(lambda x: x.split(' : ')[0] if ' : ' in x else 'Unknown')
        retrive_df['Matric'] = retrive_df['name_id'].apply(lambda x: x.split(' : ')[1] if ' : ' in x else 'Unknown')

        return retrive_df[['Name', 'Matric', 'facial_features']]
    except Exception as e:
        st.exception(f"Error in retrive_data: {e}")
        return pd.DataFrame(columns=['Name', 'Matric', 'facial_features'])



def initialize_session_state():
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    if 'show_registered_users' not in st.session_state:
        st.session_state.show_registered_users = False

def authenticate(username, password):
    stored_password = redis_client.hget('users', username)
    return stored_password and stored_password.decode() == password

# Function to check if the user is logged in
def is_user_authenticated():
    return st.session_state.get('is_authenticated', False)

def main():
    initialize_session_state()

    st.title('Attendance System')

    if st.session_state.is_authenticated:
        username = get_username()
        main_dashboard(username)
    else:
        login_menu(username='', password='', show_signup=st.session_state.show_signup)

def login_menu(username, password, show_signup):
    st.header('Login')
    
    # Retrieve the login_attempted flag
    login_attempted = st.session_state.get('login_attempted', False)
    
    # Input fields
    username_input = st.text_input('Username', value=username)
    password_input = st.text_input('Password', type='password', value=password)
    
    # Login button
    login_button = st.button('Login')
    
    if login_button:
        if authenticate(username_input, password_input):
            # Reset the login_attempted flag upon successful login
            st.session_state.login_attempted = False
            st.session_state.is_authenticated = True
            st.session_state.username = username_input
            st.success('Login successful!')
            main_dashboard(username_input)
        else:
            # Set login_attempted to True if login fails
            st.session_state.login_attempted = True
            st.warning('Incorrect username or password. Please try again.')

    if not st.session_state.show_signup:
        st.markdown('Don\'t have an account?')
        if st.button('Create new account'):
            st.session_state.show_signup = True
    else:
        signup_menu()

    # Display warning only if login attempt was made

def signup_menu():
    st.header('Sign Up')
    new_username = st.text_input('New Username')
    new_password = st.text_input('New Password', type='password')
    confirm_password = st.text_input('Confirm Password', type='password')
    submit_button = st.button('Submit')

    if submit_button:
        if new_password == confirm_password:
            redis_client.hset('users', new_username, new_password)
            st.success('Signup successful! Please log in.')
            st.session_state.show_signup = False
        else:
            st.error('Passwords do not match. Please try again.')


def get_username():
    # Replace this with the logic to get the username after authentication
    # For example, if you store the username in the session state
    return st.session_state.username

def main_dashboard(username):
    st.sidebar.title('Menu')

    # Check if the user is authenticated before showing navigation links
    if is_user_authenticated():
        selected_page = st.sidebar.radio('', ['Attendance', 'Registration', 'Report', 'Logout'])

        if selected_page == 'Attendance':
            show_real_time_prediction(username)  # Pass the username to the function
        elif selected_page == 'Registration':
            show_registration_form()
        elif selected_page == 'Report':
            show_report(username)  # Pass the username to the function
        elif selected_page == 'Logout':
            st.session_state.is_authenticated = False
            st.sidebar.empty()
            st.success('Logout successful!')
            st.experimental_rerun()

def show_real_time_prediction(username):
    
    st.text('please clear the previous logs before start every session')
    
    # Retrieve data from Redis DB
    redis_face_db = retrive_data(name='academy:register')

    # Initialize RealTimePred class with the username
    realtimepred = RealTimePred(username)
    log_name = f'attendance:logs:{username}'

    def load_logs(name, end=-1):
        logs_list = redis_client.lrange(name, start=0, end=end)
        return logs_list

    logs = load_logs(name=log_name)
    table_placeholder = st.empty()
    if st.button('Clear Logs'):
        # Clear logs in Redis
        redis_client.delete(log_name)  # Delete logs from Redis
        st.success('Logs cleared successfully!')

        # Clear logs-related session state
        st.session_state.logged_matrics = set()  # or any other session state variables related to logs
        
    st.subheader('Take your Attendance')
    def video_frame_callback(frame, username=username):  # Use lambda to capture the username
        img = frame.to_ndarray(format="bgr24")  # 3-dimensional numpy array

        # Operation that you can perform on the array
        pred_img = realtimepred.face_prediction(img, redis_face_db,
                                                'facial_features', ['Name', 'Matric'], thresh=0.5)

        # Display the processed image
        st.image(pred_img, channels="BGR")

        # Initialize a dictionary to keep track of attendance recording status for each person
        attendance_recorded_dict = {}

        # Check if a person is detected and the name is not "Unknown"
        if realtimepred.logs['name']:
            for i, person_name in enumerate(realtimepred.logs['name']):
                if person_name != 'Unknown':
                    matric = realtimepred.logs['matric'][i]
                    current_time = realtimepred.logs['current_time'][i]

                    # Display detected person's information
                    st.success(f"Person Detected: {person_name} - Matric: {matric}")
                    st.info(f"Detection Time: {current_time}")

                    # Record attendance if not already recorded
                    if person_name not in attendance_recorded_dict or not attendance_recorded_dict[person_name]:
                        # Add the attendance to logs
                        redis_client.lpush(f'attendance:logs:{username}', f"{person_name} : {matric} : {current_time}")
                        st.success(f"Attendance recorded for {person_name} at {current_time}")
                        attendance_recorded_dict[person_name] = True  # Set the flag to True to avoid duplicate attendance recording

        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

    # Streamlit webrtc component
    webrtc_streamer(key="realtimePrediction", video_frame_callback=lambda frame: video_frame_callback(frame, username=username))


def delete_user(matric_id):
    redis_client.hdel('academy:register', matric_id)




def show_registration_form():
    st.subheader('Registration Form')

    # init registration form
    registration_form = RegistrationForm()

    # Step-1: Collect person name and role
    # form
    person_name = st.text_input(label='Name', placeholder='First & Last Name')
    matric = st.text_input(label='Matric ID', placeholder='B*********')

    # Display a button to show/hide registered users
    
    # step-2: Collect facial embedding of that person
    def video_callback_func(frame):
        img = frame.to_ndarray(format='bgr24')  # 3d array bgr
        reg_img, embedding = registration_form.get_embedding(img)
        # two-step process
        # 1st step save data into local computer txt
        if embedding is not None:
            with open('face_embedding.txt', mode='ab') as f:
                np.savetxt(f, embedding)

        return av.VideoFrame.from_ndarray(reg_img, format='bgr24')

    webrtc_streamer(key='registration', video_frame_callback=video_callback_func)

    # step-3: save the data in redis database
    if st.button('Submit'):
        if not re.match(r'^[A-Z]\d{9}$', matric):
            st.warning("Please enter a valid matric ID in the format: B031920507")
        else:
            # Handle the input and validation
            return_val = registration_form.save_data_in_redis_db(person_name, matric)
            if return_val == True:
                st.success(f"{person_name} registered successfully")
            elif return_val == 'name_false':
                st.error('Please enter the name: Name cannot be empty or contain only spaces')
            elif return_val == 'file_false':
                st.error('face_embedding.txt is not found. Please refresh the page and execute again.')

    if st.button('Registered Users'):
        st.session_state.show_registered_users = not st.session_state.show_registered_users

    if st.session_state.show_registered_users:
        # Show the table of registered users
        if 'registered_users_df' not in st.session_state:
            st.session_state.registered_users_df = retrive_data(name='academy:register')

        st.table(st.session_state.registered_users_df)



        



def show_report(username):
    st.subheader('Report')

    log_name = f'attendance:logs:{username}'

    def load_logs(name, end=-1):
        logs_list = redis_client.lrange(name, start=0, end=end)
        return logs_list

    logs = load_logs(name=log_name)

    # Deduplicate logs based on 'Name' column and time difference
    deduplicated_logs = []
    last_recorded_time = {}  # Dictionary to store the last recorded time for each person

    for log in reversed(logs):  # Reverse the order to process entries in descending order (most recent first)
        entry = log.decode('utf-8').split(' : ')
        current_time = datetime.strptime(entry[2], '%Y-%m-%d %H:%M:%S.%f')
        person_name = entry[0]

        if (
            person_name not in last_recorded_time
            or (current_time - last_recorded_time[person_name]).seconds >= 10
        ):
            deduplicated_logs.append(entry)
            last_recorded_time[person_name] = current_time

    deduplicated_logs_df = pd.DataFrame(reversed(deduplicated_logs), columns=['Name', 'Matric', 'Time'])

    # Use st.empty() to clear the existing table
    table_placeholder = st.empty()

    # Display logs in a table
    table_placeholder.table(deduplicated_logs_df)

    if st.button('Download Logs CSV'):
        csv_data = deduplicated_logs_df.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="logs.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    if st.button('Clear Logs'):
        # Clear logs in Redis
        redis_client.delete(log_name)  # Delete logs from Redis
        st.success('Logs cleared successfully!')

        # Clear logs-related session state
        st.session_state.logged_matrics = set()  # or any other session state variables related to logs
        table_placeholder.table(pd.DataFrame(columns=['Name', 'Matric', 'Time']))  # Recreate the table with an empty DataFrame

if __name__ == '__main__':
    initialize_session_state()
    main()
