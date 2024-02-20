import shutil

from peewee import *

db = SqliteDatabase('chatbot.db')


class BaseModel(Model):
    class Meta:
        database = db  # This model uses the "chatbot.db" database.


class ChatHistory(BaseModel):
    id = AutoField(primary_key=True)
    query = TextField()
    model = TextField()
    answer = TextField()
    millisecs = IntegerField()


class ChatBenchmark(BaseModel):
    id = AutoField(primary_key=True)
    query = TextField()
    model_a = TextField()
    model_b = TextField()
    answer_a = TextField()
    answer_b = TextField()
    time_a = IntegerField()
    time_b = IntegerField()
    winner = TextField()
    judge = TextField()


class ProjectsQueryFeedBack(BaseModel):
    id = AutoField(primary_key=True)
    query = TextField()
    answer = TextField()
    feedback = TextField()


# Function to append data to the backup file
def append_to_backup():
    backup_filename = 'backup_chatbot.db'
    with open('chatbot.db', 'rb') as src, open(backup_filename, 'ab') as dest:
        shutil.copyfileobj(src, dest)


append_to_backup()
db.connect()
db.create_tables([ChatHistory])
db.create_tables([ChatBenchmark])
db.create_tables([ProjectsQueryFeedBack])
