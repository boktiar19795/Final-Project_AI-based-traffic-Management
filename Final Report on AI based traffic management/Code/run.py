import os
from utilities import Database, elight, Yolo

dbPath = os.path.join(os.getcwd(), "db", "historical.db")
database = Database(dbPath)

app = elight(database)

if __name__ == "__main__":
    # TODO: Uncomment two lines below for filling the database randomly
    # database.empty()
    # database.fillRandom("3 months", "1 hour")

    # TODO: Uncomment two lines below for finetuning the Yolo model
    # yolo = Yolo('yolov8x') # ['yolov8n', 'yolov8m', 'yolov8l', 'yolov8x', 'yolov8s']
    # yolo.finetune(os.path.join(os.getcwd(), 'ds', 'data.yaml'), epochs=100)

    # TODO: Comment one line below if any of TODOs above are working
    app.run(frameRate=8) # for normal processing
    # app.run(mode='threads') # for multithread processing [[Not Available]]



    