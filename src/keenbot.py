

import streamlit as st
import boto3
import pickle
import json
from sqlalchemy import MetaData, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

# Initialize AWS Bedrock client
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest."

# SQLAlchemy setup
Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    system_prompt = Column(String, nullable=False)
    model_id = Column(Integer, ForeignKey('models.id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    message = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Model(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    configuration = Column(String, nullable=False)

# Initialize database connection
@st.cache_resource
def init_connection():
    conn = st.connection('keenbot_db', type='sql')
    # Define metadata for the table
    #metadata = MetaData()

    # Define the table with appropriate columns
    #Table('conversations', metadata,
    #      Column('id', Integer, primary_key=True),
    #      Column('name', String, nullable=False),
    #      Column('system_prompt', String, nullable=False),
    #      Column('model_id', Integer, ForeignKey('models.id')),
    #      Column('created_at', DateTime(timezone=True), server_default=func.now())
    #      )

    # Create the table if it doesn't exist
    Base.metadata.create_all(conn.engine)
    return conn

# Function to add or edit a model
def add_or_edit_model(conn, name, configuration, model_id=None):
    if model_id:
        model = conn.session.query(Model).filter_by(id=model_id)[0]
        model.name = name
        model.configuration = configuration
        conn.update(model)
    else:
        new_model = Model(name=name, configuration=configuration)
        conn.add(new_model)

# Function to get all models
def get_models(conn):
    return conn.session.query(Model)

# Function to get model by id
def get_model(conn, model_id):
    result = conn.session.query(Model).filter_by(id=model_id)
    if result:
        model = result[0]
        return model.name, model.configuration
    return None, None

# Function to create a new conversation
def create_conversation(conn, name, system_prompt, model_id):
    new_conversation = Conversation(name=name, system_prompt=system_prompt, model_id=model_id)
    conn.add(new_conversation)
    return new_conversation.id

# Function to save a message
def save_message(conn, conversation_id, message):
    serialized_message = serialize_message(message)
    new_message = Message(conversation_id=conversation_id, message=serialized_message)
    conn.add(new_message)

# Function to load messages for a conversation
def load_messages(conn, conversation_id):
    messages = conn.session.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at)
    return [deserialize_message(message.message) for message in messages]

# Function to get all conversations
def get_conversations(conn):
    return conn.session.query(Conversation).order_by(Conversation.created_at.desc())

# Function to get system prompt and model for a conversation
def get_conversation_details(conn, conversation_id):
    result = conn.session.query(Conversation).filter_by(id=conversation_id)
    if result:
        conversation = result[0]
        return conversation.system_prompt, conversation.model_id
    return DEFAULT_SYSTEM_PROMPT, None

# Serialization and deserialization functions (unchanged)
def serialize_message(message):
    return pickle.dumps(message)

def deserialize_message(serialized_message):
    return pickle.loads(serialized_message)

# Function to create a new conversation chain
def create_conversation_chain(conn, conversation_id=None):
    memory = ConversationBufferMemory(return_messages=True)
    if conversation_id:
        messages = load_messages(conn, conversation_id)
        memory.chat_memory.messages = messages
        system_prompt, model_id = get_conversation_details(conn, conversation_id)
        if model_id:
            model_name, model_config = get_model(conn, model_id)
            llm = Bedrock(
                model_id=model_name,
                client=bedrock_client,
                model_kwargs=json.loads(model_config)
            )
        else:
            llm = Bedrock(
                model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
                client=bedrock_client,
                model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.7}
            )
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        llm = Bedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            client=bedrock_client,
            model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.7}
        )
    
    prompt_template = f"{system_prompt}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nClaude:"
    
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=prompt_template
    )
    
    return ConversationChain(
        llm=llm,
        memory=memory,
        prompt=PROMPT
    )

# Main Streamlit app
def main():
    # Initialize database connection
    conn = init_connection()

    # Initialize session state
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None

    # Sidebar for conversation selection and model management
    st.sidebar.title("Conversations and Models")

    # Model management
    st.sidebar.subheader("Model Management")
    models = get_models(conn)
    model_names = [model.name for model in models]
    selected_model = st.sidebar.selectbox("Select Model", [""] + model_names)

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Add Model"):
        st.session_state.add_model = True
    if col2.button("Edit Model"):
        if selected_model:
            st.session_state.edit_model = True
        else:
            st.sidebar.warning("Please select a model to edit.")

    # Add Model dialog
    if 'add_model' in st.session_state and st.session_state.add_model:
        with st.sidebar.form("add_model_form"):
            st.subheader("Add New Model")
            new_model_name = st.text_input("Model Name")
            new_model_config = st.text_area("Model Configuration (JSON)")
            submitted = st.form_submit_button("Proceed")
            if submitted:
                try:
                    json.loads(new_model_config)  # Validate JSON
                    add_or_edit_model(conn, new_model_name, new_model_config)
                    st.success("Model added successfully!")
                    st.session_state.add_model = False
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON configuration. Please check and try again.")
            if st.form_submit_button("Cancel"):
                st.session_state.add_model = False
                st.rerun()

    # Edit Model dialog
    if 'edit_model' in st.session_state and st.session_state.edit_model:
        with st.sidebar.form("edit_model_form"):
            st.subheader("Edit Model")
            model_to_edit = next(model for model in models if model.name == selected_model)
            edit_model_name = st.text_input("Model Name", value=model_to_edit.name)
            edit_model_config = st.text_area("Model Configuration (JSON)", value=model_to_edit.configuration)
            submitted = st.form_submit_button("Proceed")
            if submitted:
                try:
                    json.loads(edit_model_config)  # Validate JSON
                    add_or_edit_model(conn, edit_model_name, edit_model_config, model_to_edit.id)
                    st.success("Model updated successfully!")
                    st.session_state.edit_model = False
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON configuration. Please check and try again.")
            if st.form_submit_button("Cancel"):
                st.session_state.edit_model = False
                st.rerun()

    # New conversation creation
    st.sidebar.subheader("Create New Conversation")
    new_conversation = st.sidebar.text_input("New conversation label")
    new_system_prompt = st.sidebar.text_area("System prompt for new conversation", DEFAULT_SYSTEM_PROMPT)
    new_conversation_model = st.sidebar.selectbox("Model for new conversation", model_names)

    if st.sidebar.button("Create New Conversation"):
        if new_conversation and new_conversation_model:
            model_id = next(model.id for model in models if model.name == new_conversation_model)
            conversation_id = create_conversation(conn, new_conversation, new_system_prompt, model_id)
            st.session_state.current_conversation = conversation_id
            st.rerun()
        else:
            st.sidebar.warning("Please provide a conversation name and select a model.")

    # Existing conversation selection
    st.sidebar.subheader("Select Existing Conversation")
    conversations = get_conversations(conn)
    conversation_options = ["New Chat"] + [f"{conv.id}: {conv.name}" for conv in conversations]
    selected_conversation = st.sidebar.selectbox("Select a conversation", conversation_options)

    if selected_conversation != "New Chat":
        conversation_id = int(selected_conversation.split(":")[0])
        if st.session_state.current_conversation != conversation_id:
            st.session_state.current_conversation = conversation_id

    # Main chat interface
    st.title("Claude 3.5 Sonnet Chatbot")

    # Display current system prompt and model
    if st.session_state.current_conversation:
        system_prompt, model_id = get_conversation_details(conn, st.session_state.current_conversation)
        model_name, _ = get_model(conn, model_id) if model_id else (None, None)
        model_name = model_name or "Default"
        st.info(f"Current System Prompt: {system_prompt}")
        st.info(f"Current Model: {model_name}")

    # Display conversation history
    if st.session_state.current_conversation:
        chain = create_conversation_chain(conn, st.session_state.current_conversation)
        for message in chain.memory.chat_memory.messages:
            with st.chat_message(message.type):
                st.write(message.content)

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Get or create conversation chain
        if not st.session_state.current_conversation:
            st.warning("Please create or select a conversation before chatting.")
        else:
            chain = create_conversation_chain(conn, st.session_state.current_conversation)
            
            # Get Claude's response
            response = chain.predict(input=user_input)

            # Display user message and Claude's response
            with st.chat_message("human"):
                st.write(user_input)
            with st.chat_message("ai"):
                st.write(response)

            # Save messages to database
            save_message(conn, st.session_state.current_conversation, HumanMessage(content=user_input))
            save_message(conn, st.session_state.current_conversation, AIMessage(content=response))

# Run the Streamlit app
if __name__ == "__main__":
    main()
