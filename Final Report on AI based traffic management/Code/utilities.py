import os
import copy
import random
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any
import sqlite3
import tempfile


import cv2
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO


from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings

warnings.filterwarnings("ignore")


class Database:
    '''
    A class to interact with the SQLite database for vehicle and light duration data.
    '''

    def __init__(self, dbName: str):
        self.dbName = dbName
        self._createTables()

    def _createTables(self):
        '''
        Create the detections and lights tables if they do not already exist.
        '''
        conn = sqlite3.connect(self.dbName)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            time TEXT,
            side TEXT,
            car INTEGER,
            bus INTEGER,
            truck INTEGER,
            motorcycle INTEGER,
            bicycle INTEGER,
            total INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS lights (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            detectionID INTEGER,
            green INTEGER,
            FOREIGN KEY (detectionID) REFERENCES detections (ID)
        )
        ''')
        
        conn.commit()
        conn.close()

    def empty(self) -> None:
        '''
        Empty all tables in the SQLite database.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        conn = sqlite3.connect(self.dbName)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            cursor.execute(f"DELETE FROM {table[0]};")
        
        conn.commit()
        conn.close()

    def fillRandom(self, period: str, timeStep: str) -> None:
        '''
        Fills the database with random data over a specified period and time step.

        Parameters
        ----------
        period : str
            Period over which to generate data (e.g., '1 day', '1 week', '1 month', '1 year').
        timeStep : str
            Time step for data generation (e.g., '30 seconds', '2 minutes', '1 hour', '1 day').

        '''
        conn = sqlite3.connect(self.dbName)
        cursor = conn.cursor()

        periodDelta = self.parseTime(period)
        timeStepDelta = self.parseTime(timeStep)

        startDate = datetime.now() - periodDelta
        endDate = datetime.now()

        currentTime = startDate

        sides = ['North', 'South', 'East', 'West']

        while currentTime <= endDate:
            dateStr = currentTime.strftime('%Y-%m-%d')
            timeStr = currentTime.strftime('%H:%M:%S')

            for side in sides:
                carCount = random.randint(0, 20)
                busCount = random.randint(0, 5)
                truckCount = random.randint(0, 10)
                motorcycleCount = random.randint(0, 15)
                bicycleCount = random.randint(0, 10)
                totalCount = carCount + busCount + truckCount + motorcycleCount + bicycleCount

                green = int(10 + 0.5 * totalCount)

                cursor.execute('''
                INSERT INTO detections (date, time, side, car, bus, truck, motorcycle, bicycle, total)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (dateStr, timeStr, side, carCount, busCount, truckCount, motorcycleCount, bicycleCount, totalCount))
                
                detectionID = cursor.lastrowid
                
                cursor.execute('''
                INSERT INTO lights (detectionID, green)
                VALUES (?, ?)
                ''', (detectionID, green))

            currentTime += timeStepDelta

        conn.commit()
        conn.close()

    def parseTime(self, timeStr: str) -> timedelta:
        '''
        Parses a time string to a timedelta object.

        Parameters
        ----------
        timeStr : str
            Time string (e.g., '1 day', '2 hours', '30 minutes').

        Returns
        -------
        timedelta
            Corresponding timedelta object.
        '''
        timeUnits = {
            'second': timedelta(seconds=1),
            'minute': timedelta(minutes=1),
            'hour': timedelta(hours=1),
            'day': timedelta(days=1),
            'week': timedelta(weeks=1),
            'month': timedelta(days=30),
            'year': timedelta(days=365) 
        }

        quantity, unit = timeStr.split()
        quantity = int(quantity)
        unit = unit.rstrip('s') 
        return quantity * timeUnits[unit]

    @staticmethod
    def insertVehicles(dbName: str, date: str, time: str, side: str, car: int, bus: int, truck: int, motorcycle: int, bicycle: int, total: int) -> int:
        '''
        Insert a new record into the detections table.

        Parameters
        ----------
        dbName : str
            Name of the SQLite database file.
        date : str
            Date of the record.
        time : str
            Time of the record.
        side : str
            Side of the road or relevant identifier.
        car : int
            Number of cars.
        bus : int
            Number of buses.
        truck : int
            Number of trucks.
        motorcycle : int
            Number of motorcycles.
        bicycle : int
            Number of bicycles.
        total : int
            Total number of vehicles.

        Returns
        -------
        int
            ID of the newly inserted record.
        '''
        conn = sqlite3.connect(dbName)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO detections (date, time, side, car, bus, truck, motorcycle, bicycle, total)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, time, side, car, bus, truck, motorcycle, bicycle, total))
        
        conn.commit()
        lastRowId = cursor.lastrowid
        conn.close()
        return lastRowId

    @staticmethod
    def insertLights(dbName: str, detectionID: int, green: int):
        '''
        Insert a new record into the lights table.

        Parameters
        ----------
        dbName : str
            Name of the SQLite database file.
        detectionID : int
            ID of the related detection record.
        green : int
            Green light duration.
        '''
        conn = sqlite3.connect(dbName)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO lights (detectionID, green)
        VALUES (?, ?)
        ''', (detectionID, green))
        
        conn.commit()
        conn.close()

    @staticmethod
    def readVehicles(dbName: str, date: str, time: str, side: str) -> pd.DataFrame:
        '''
        Read vehicle records from the detections table based on date, time, and side.

        Parameters
        ----------
        dbName : str
            Name of the SQLite database file.
        date : str
            The date of the record to retrieve.
        time : str
            The time of the record to retrieve.
        side : str
            The side of the record to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the records that match the date, time, and side.
        '''
        conn = sqlite3.connect(dbName)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT car, bus, truck, motorcycle, bicycle 
        FROM detections 
        WHERE date = ? AND time = ? AND side = ?
        ''', (date, time, side))
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        conn.close()

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def readLights(dbName: str) -> pd.DataFrame:
        '''
        Read all records from the lights table.

        Parameters
        ----------
        dbName : str
            Name of the SQLite database file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all records from the lights table.
        '''
        conn = sqlite3.connect(dbName)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM lights')
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        conn.close()

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def fetchAllDetectionData(dbName: str) -> pd.DataFrame:
        '''
        Fetch all vehicle detection data.

        Parameters
        ----------
        dbName : str
            Name of the SQLite database file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all records from the detections table.
        '''
        conn = sqlite3.connect(dbName)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM detections')
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        conn.close()

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def fetchAllLightData(dbName: str) -> pd.DataFrame:
        '''
        Fetch all light duration data.

        Parameters
        ----------
        dbName : str
            Name of the SQLite database file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all records from the lights table.
        '''
        conn = sqlite3.connect(dbName)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM lights')
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        conn.close()

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def fetchJoinedData(dbName: str) -> pd.DataFrame:
        '''
        Fetch joined vehicle detection and light duration data.

        Parameters
        ----------
        dbName : str
            Name of the SQLite database file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing joined data from detections and lights tables.
        '''
        conn = sqlite3.connect(dbName)
        query = '''
        SELECT d.date, d.time, d.side, d.car, d.bus, d.truck, d.motorcycle, d.bicycle, d.total, l.green
        FROM detections d
        LEFT JOIN lights l ON d.ID = l.detectionID
        '''
        data = pd.read_sql_query(query, conn)
        conn.close()

        return data


class Yolo:
    '''
    A class to encapsulate YOLOv8 object detection and tracking.

    Attributes
    ----------
    model : YOLO
        YOLOv8 model instance for object detection.
    vehicleTypes : List[str]
        List of vehicle types to detect.
    lastSaveTime : datetime
        Timestamp of the last data save operation.
    saveTimeout : int
        Timeout duration in seconds for saving data into the database.
    '''

    def __init__(self, modelName: str):
        self.modelName = modelName
        modelPaths = {
            'yolov8n': [os.path.join(os.getcwd(), 'md', 'yolov8n.pt'), os.path.join(os.getcwd(), 'pt', 'yolov8sFinetuned.pt')],
            'yolov8m': [os.path.join(os.getcwd(), 'md', 'yolov8m.pt'), os.path.join(os.getcwd(), 'pt', 'yolov8sFinetuned.pt')],
            'yolov8l': [os.path.join(os.getcwd(), 'md', 'yolov8l.pt'), os.path.join(os.getcwd(), 'pt', 'yolov8sFinetuned.pt')],
            'yolov8x': [os.path.join(os.getcwd(), 'md', 'yolov8x.pt'), os.path.join(os.getcwd(), 'pt', 'yolov8sFinetuned.pt')],
            'yolov8s': [os.path.join(os.getcwd(), 'md', 'yolov8s.pt'), os.path.join(os.getcwd(), 'pt', 'yolov8sFinetuned.pt')],
        }
        self.model = YOLO(modelPaths[self.modelName][0])
        self.finetunedModel = YOLO(modelPaths[self.modelName][1])
        self.classMap = {
            'Aerial_Apparatus': 'fire truck',
            'Ambulance_Bus': 'ambulance',
            'Bariatric_Ambulance': 'ambulance',
            'Emergency_Ambulance': 'ambulance',
            'Motorcycle_Ambulance': 'ambulance',
            'Conventional_Fire_Engine': 'fire truck',
            'Fire_Command_Vehicle': 'fire truck',
            'Fire_Investigation_Unit': 'fire truck',
            'Platform_Truck': 'truck',
            'Police_Bus': 'police',
            'Police_Crossover': 'police',
            'Police_Hatchback': 'police',
            'Police_Motorcycle': 'police',
            'Police_Off_Road_Vehicle': 'police',
            'Police_Pickup_Truck': 'police',
            'Police_SUV': 'police',
            'Police_Sedan': 'police',
            'Police_Van': 'police',
            'Rescue_Vehicle': 'fire truck',
            'SWAT_Vehicle': 'police',
            'Water_Tender': 'fire truck',
            'Wildland_Fire_Engine': 'fire truck',
            'car': 'car',
            'truck': 'truck',
            'bus': 'bus',
            'motorcycle': 'motorcycle',
            'bicycle': 'bicycle',
        }
        self.colorMap = {
            'car': (0, 0, 255),          # Red
            'bus': (0, 255, 0),          # Green
            'truck': (255, 0, 0),        # Blue
            'motorcycle': (0, 255, 255), # Cyan
            'bicycle': (255, 255, 0),    # Yellow
            'fire truck': (0, 128, 255), # Orange
            'ambulance': (255, 0, 255),  # Magenta
            'police': (128, 0, 128)      # Purple
        }
        self.vehicleTypes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'fire truck', 'ambulance', 'police']

    def iou(self, boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        boxA : list of float
            The first bounding box, represented as a list of four floats: [x1, y1, x2, y2].
        boxB : list of float
            The second bounding box, represented as a list of four floats: [x1, y1, x2, y2].

        Returns
        -------
        float
            The IoU of the two bounding boxes. IoU is a float between 0 and 1, where 1 represents perfect overlap and 0 represents no overlap.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxB[3], boxA[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def visualizeTracking(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        '''
        Visualize object tracking on a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Input frame for visualization.

        Returns
        -------
        tuple
            A tuple containing:
            - The frame with visualized tracking (numpy.ndarray).
            - A dictionary with counts of each vehicle type (Dict[str, int]).
        '''
        resultsPrimary = self.model(frame)
        classNamesPrimary = self.model.names
        boxesPrimary = resultsPrimary[0].boxes.xyxy.tolist()
        classIdsPrimary = resultsPrimary[0].boxes.cls.tolist()
        confidencesPrimary = resultsPrimary[0].boxes.conf.tolist()

        if self.finetunedModel:
            resultsFinetuned = self.finetunedModel(frame)
            classNamesFinetuned = self.finetunedModel.names
            boxesFinetuned = resultsFinetuned[0].boxes.xyxy.tolist()
            classIdsFinetuned = resultsFinetuned[0].boxes.cls.tolist()
            confidencesFinetuned = resultsFinetuned[0].boxes.conf.tolist()
        else:
            boxesFinetuned = []
            classIdsFinetuned = []
            confidencesFinetuned = []

        combinedBoxes = []
        combinedClassIds = []
        combinedConfidences = []

        for boxF, classIdF, confidenceF in zip(boxesFinetuned, classIdsFinetuned, confidencesFinetuned):
            for i, boxP in enumerate(boxesPrimary):
                if self.iou(boxF, boxP) > 0.5: 
                    boxesPrimary.pop(i)
                    classIdsPrimary.pop(i)
                    confidencesPrimary.pop(i)
                    break

            combinedBoxes.append(boxF)
            combinedClassIds.append(classIdF)
            combinedConfidences.append(confidenceF)

        combinedBoxes.extend(boxesPrimary)
        combinedClassIds.extend(classIdsPrimary)
        combinedConfidences.extend(confidencesPrimary)

        vehicleCounts = {vehicle: 0 for vehicle in self.vehicleTypes}
        vehicleCounts.update({'total': 0})

        for box, classId, confidence in zip(combinedBoxes, combinedClassIds, combinedConfidences):
            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box)
                className = (classNamesPrimary if classId in classIdsPrimary else classNamesFinetuned)[int(classId)]
                mappedClassName = self.classMap.get(className, None)
                if mappedClassName and mappedClassName in vehicleCounts:
                    vehicleCounts[mappedClassName] += 1
                    vehicleCounts['total'] += 1

                    color = self.colorMap.get(mappedClassName, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{mappedClassName}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        yOffset = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontThickness = 2
        textColor = (255, 255, 255) 
        backgroundColor = (0, 0, 0) 

        for vehicle, count in vehicleCounts.items():
            text = f"{vehicle}: {count}"
            (textWidth, textHeight), baseline = cv2.getTextSize(text, font, fontScale, fontThickness)
            
            cv2.rectangle(frame, (10, yOffset - textHeight - 10), (10 + textWidth, yOffset + baseline), backgroundColor, thickness=cv2.FILLED)
            cv2.putText(frame, text, (10, yOffset), font, fontScale, textColor, fontThickness)
            
            yOffset += textHeight + baseline + 10

        return frame, vehicleCounts

    
    def finetune(self, dataPath: str, epochs: int = 10, batchSize: int = 16, imageSize: int = 640, ):
        '''
        Fine-tune the YOLO model on a custom dataset.

        Parameters
        ----------
        dataPath : str
            Path to the dataset for fine-tuning.
        epochs : int, optional
            Number of epochs to train the model, by default 10.
        batchSize : int, optional
            Batch size for training, by default 16.
        '''
        self.model.train(data=dataPath, epochs=epochs, imgsz=imageSize, batch=batchSize)
        self.model.save(f"{self.modelName}Finetuned.pt")


class Halo:
    '''
    A class to handle time series prediction of vehicle counts using various models.

    Attributes
    ----------
    models : Dict[str, Any]
        Dictionary of models.
    fittedModels : Dict[str, Any]
        Dictionary of fitted models.
    performanceMetrics : Dict[str, Dict[str, float]]
        Dictionary to store overall performance metrics for fitted models.
    '''

    def __init__(self) -> None:
        self.models = {
            'randomForest': MultiOutputRegressor(RandomForestRegressor()),
            'svr': MultiOutputRegressor(SVR()),
            'gradientBoosting': MultiOutputRegressor(GradientBoostingRegressor()),
            'knn': MultiOutputRegressor(KNeighborsRegressor()),
            'extraTrees': MultiOutputRegressor(ExtraTreesRegressor())
        }
        self.fittedModels = dict()
        self.performanceMetrics = dict()

    def preprocess(self, model: Any) -> Pipeline:
        '''
        Create a preprocessing and modeling pipeline.

        Parameters
        ----------
        model : Any
            The machine learning model to include in the pipeline.

        Returns
        -------
        Pipeline
            A pipeline with preprocessing and the specified model.
        '''
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['datetime']), 
                ('cat', OneHotEncoder(), ['side'])
            ]
        )

        return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    def fitModel(self, modelName: str, data: pd.DataFrame) -> Dict[str, float]:
        '''
        Fit the specified model to the data and calculate overall performance metrics.

        Parameters
        ----------
        modelName : str
            Name of the model to use.
        data : pd.DataFrame
            DataFrame containing the time series data.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the performance metrics for the fitted model.
        '''
        
        if modelName not in self.models:
            raise ValueError(f"Model {modelName} is not available.")
        
        data = data.copy()
        data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data.drop(['date', 'time'], axis=1, inplace=True)
        data = data.sort_values(by='datetime')

        if modelName in ['randomForest', 'svr', 'gradientBoosting', 'knn', 'extraTrees']:
            model = self.models[modelName]
            pipeline = self.preprocess(model)

            X = data[['datetime', 'side']]
            X.loc[:, 'datetime'] = X['datetime'].map(pd.Timestamp.timestamp)
            y = data[['car', 'bus', 'truck', 'bicycle', 'motorcycle']]

            fittedModel = pipeline.fit(X, y)
            self.fittedModels[modelName] = fittedModel
            
            predictions = pipeline.predict(X)
            actuals = y
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions, multioutput='uniform_average')
            
            return {
                'model': [modelName],
                'mse': [mse],
                'mae': [mae],
                'r2': [r2]
            }

    def predict(self, modelName: str, date: str, time: str, side: str) -> Dict[str, Any]:
        '''
        Predict the number of vehicles using the specified model.

        Parameters
        ----------
        modelName : str
            Name of the model to use.
        date : str
            Date for prediction in 'YYYY-MM-DD' format.
        time : str
            Time for prediction in 'HH:MM:SS' format.
        side : str
            Side of the road or relevant identifier.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the predicted number of vehicles for each type and total.
        '''
        if modelName not in self.fittedModels:
            raise ValueError(f"Model {modelName} has not been fitted yet.")

        datetime = date + ' ' + time
        datetime = pd.to_datetime(datetime)
        timestamp = datetime.timestamp()

        prediction = {'car': None, 'bus': None, 'truck': None, 'bicycle': None, 'motorcycle': None, 'total': None}

        if modelName in ['randomForest', 'svr', 'gradientBoosting', 'knn', 'extraTrees']:
            pipeline = self.fittedModels[modelName]
            xNew = pd.DataFrame([[timestamp, side]], columns=['datetime', 'side'])
            yPred = pipeline.predict(xNew)
            prediction.update({
                'car': round(yPred[0][0]),
                'bus': round(yPred[0][1]),
                'truck': round(yPred[0][2]),
                'bicycle': round(yPred[0][3]),
                'motorcycle': round(yPred[0][4]),
                'total': sum([round(yPred[0][i]) for i in range(5)])
            })

        return prediction


class Jelo:
    '''
    A class to determine green light durations using vehicle counts.

    Attributes
    ----------
    models : Dict[str, Any]
        Dictionary of models.
    fittedModels : Dict[str, Pipeline]
        Dictionary of fitted model pipelines.
    '''

    def __init__(self) -> None:
        self.models = {
            'decisionTreeRegressor': DecisionTreeRegressor(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elasticNet': ElasticNet(),
            'bayesianRidge': BayesianRidge()
        }
        self.fittedModels = dict()

    def preprocess(self, model: Any) -> Pipeline:
        '''
        Create a preprocessing and modeling pipeline.

        Parameters
        ----------
        model : Any
            The machine learning model to include in the pipeline.

        Returns
        -------
        Pipeline
            A pipeline with preprocessing and the specified model.
        '''
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['car', 'bus', 'truck', 'bicycle', 'motorcycle', 'total', 'timestamp']),
                ('cat', OneHotEncoder(), ['side'])
            ]
        )

        return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    def fitModel(self, modelName: str, data: pd.DataFrame) -> Dict[str, float]:
        '''
        Fit the specified model to the data and return performance metrics.

        Parameters
        ----------
        modelName : str
            Name of the model to use.
        data : pd.DataFrame
            DataFrame containing the time series data.

        Returns
        -------
        Dict[str, float]
            Dictionary containing performance metrics.
        '''
        if modelName not in self.models:
            raise ValueError(f"Model {modelName} is not available.")

        model = self.models[modelName]

        data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data = data.sort_values(by='datetime')
        data['timestamp'] = data['datetime'].map(pd.Timestamp.timestamp)

        X = data[['side', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'total', 'timestamp']]
        y = data['green']

        xTrain, xVal, yTrain, yVal = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = self.preprocess(model)
        pipeline.fit(xTrain, yTrain)

        yPred = pipeline.predict(xVal)

        mse = mean_squared_error(yVal, yPred)
        mae = mean_absolute_error(yVal, yPred)
        r2 = r2_score(yVal, yPred)

        self.fittedModels[modelName] = pipeline

        return {
            'model': [modelName],
            'mse': [mse],
            'mae': [mae],
            'r2': [r2]
        }

    def predict(self, modelName: str, data: pd.DataFrame) -> Dict[str, int]:
        '''
        Predict the red or green light duration using the specified model.

        Parameters
        ----------
        modelName : str
            Name of the model to use for prediction.
        data : pd.DataFrame
            DataFrame containing the features for prediction.

        Returns
        -------
        Dict[str, int]
            Predicted values for the specified light.
        '''
        if modelName not in self.fittedModels:
            raise ValueError(f"Model {modelName} has not been fitted yet.")

        pipeline = self.fittedModels[modelName]

        data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data = data.sort_values(by='datetime')
        data['timestamp'] = data['datetime'].map(pd.Timestamp.timestamp)

        X = data[['side', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'total', 'timestamp']]
        predictions = pipeline.predict(X)

        return {'red': None, 'green': int(predictions[0])}

    
class elight:
    def __init__(self, db: Database):
        self.db = db
        self.vehicles = {
            'date': [],
            'time': [],
            'side': [],
            'car': [],
            'bus': [],
            'truck': [],
            'motorcycle': [],
            'bicycle': [],
            'total': []
        }

        self.emergency = {
            'fire truck': [], 
            'ambulance': [], 
            'police': []
        }

        self.lights = {
            'date': [],
            'time': [],
            'side': [],
            'green': [],
            'red': []
        }
        self.metrics = {
            'model': [],
            'mse': [],
            'mae': [],
            'r2': []
        }

    def update(self, main: Dict[str, List[Any]], data: Dict[str, Any], date: str, time: str, side: str) -> Dict[str, List[Any]]:
        """
        Update the main dictionary with new data.

        Parameters
        ----------
        main : dict
            The main dictionary to be updated.
        data : dict
            The new data to update the main dictionary with.
        date : str
            The date of the data.
        time : str
            The time of the data.
        side : str
            The side of the data.

        Returns
        -------
        dict
            The updated main dictionary.
        """
        updated = data.copy()
        updated.update({'date': date, 'time': time, 'side': side})

        for key in main.keys():
            if isinstance(updated[key], list):
                main[key].extend(updated[key])
            else:
                main[key].append(updated[key])

                
        return main

    def run(self, frameRate: int = 6):
        '''
        Run the Streamlit application.

        Parameters
        ----------
        frameRate:
            The rate of showing processed images (6).
        '''
        st.set_page_config(page_title='Traffic Monitoring', page_icon=os.path.join(os.getcwd(), 'ast', 'traffic.png'))

        st.title('Traffic Analysis and Prediction System')

        st.header('Objective')
        st.write('''
            The primary objective of this project is to analyze traffic videos to detect and classify vehicles, 
            and to make predictions about traffic conditions using various machine learning models. 
            The system processes video feeds from different directions (e.g., North, East, South, West), 
            performs real-time object detection, and combines results from multiple models to improve prediction accuracy.
        ''')

        st.header('Components')
        st.write('''
            - **Video Upload Interface**: Allows users to upload videos from different directions for processing.
            - **Object Detection**: Utilizes YOLO (You Only Look Once) model to detect and track vehicles in video frames.
            - **Prediction Models**: Uses Halo and Jelo models to predict traffic conditions and light durations.
            - **Database Integration**: Stores results in a SQLite database, including vehicle counts and light durations.
            - **Streamlit Interface**: Provides a web interface for video upload, model selection, and result display.
        ''')

        st.header('Workflow')
        st.write('''
            - **Video Upload**: Users upload videos, which are temporarily stored for processing.
            - **Video Processing**: YOLO model detects and tracks vehicles in video frames, processed at specified intervals.
            - **Model Predictions**: Halo and Jelo models predict traffic conditions based on detected vehicles.
            - **Results Storage**: Stores results in SQLite database, including timestamps, vehicle counts, and light durations.
            - **Real-time Display**: Displays processed frames and results in real-time.
        ''')

        st.header('Technologies Used')
        st.write('''
            - **Python**: Core language for system implementation.
            - **OpenCV**: For video processing and frame handling.
            - **Streamlit**: Web framework for user interface.
            - **YOLO**: Object detection model for vehicle tracking.
            - **SQLite**: Database for storing results.
            - **Scikit-learn**: Machine learning library for prediction models.
            - **Pandas**: For data manipulation and analysis.
        ''')

        st.header('Key Files and Their Roles')
        st.write('''
            - **run.py**: Main script to run the application.
            - **utilities.py**: Contains helper functions and classes for video processing, model handling, and result updating.
            - **requirements.txt**: Lists all project dependencies.
        ''')

        st.header('Team Members')
        st.write('''
            - Member1
            - Member2
            - Member3
            - Member4
        ''')

        st.header('Date and Time')
        dateColumn, timeColumn = st.columns(2)
        with dateColumn:
            date = st.date_input('Select Date', value=datetime.now().date()).strftime('%Y-%m-%d')
            st.success(f"{date} selected successfully.")

        with timeColumn:
            time = st.time_input('Select Time', value=datetime.now().time()).strftime('%H:%M:%S')
            st.success(f"{time} selected successfully.")

        st.header('Video Uploads')
        sides = ['North', 'East', 'South', 'West']
        videoUploads = {}
        for side, column in zip(sides[:2], st.columns(2)):
            with column:
                videoUpload = st.file_uploader(f"{side} Video", type=['mp4'])
                if videoUpload:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tempFile:
                        tempFile.write(videoUpload.read())
                        videoUploads[side] = tempFile.name

        for side, column in zip(sides[2:], st.columns(2)):
            with column:
                videoUpload = st.file_uploader(f"{side} Video", type=['mp4'])
                if videoUpload:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tempFile:
                        tempFile.write(videoUpload.read())
                        videoUploads[side] = tempFile.name

        st.header('YOLO Model Selection')
        yoloModels = ['yolov8n', 'yolov8m', 'yolov8l', 'yolov8x', 'yolov8s']
        selectedYoloModel = st.selectbox('Select YOLO Model', yoloModels, index=None)
        yolo = Yolo(modelName=selectedYoloModel) if selectedYoloModel else Yolo(modelName='yolov8s')
        st.success(f"{selectedYoloModel} model selected successfully.") if selectedYoloModel else None

        detectionData = self.db.fetchAllDetectionData(self.db.dbName)
        joinedData = self.db.fetchJoinedData(self.db.dbName)

        st.header('Vehicle and Light Duration Model Selection')
        haloModels = ['randomForest', 'svr', 'gradientBoosting', 'knn', 'extraTrees']
        selectedHaloModel = st.selectbox('Select Halo Model', haloModels, index=None)
        halo = Halo()
        haloMetrics = halo.fitModel(selectedHaloModel, detectionData.drop(['total'], axis=1)) if selectedHaloModel else None
        st.success(f"{selectedHaloModel} model selected successfully.") if selectedHaloModel else None

        st.write('Halo Model Performance Metrics:')
        st.dataframe(pd.DataFrame(haloMetrics) if haloMetrics else pd.DataFrame(self.metrics))

        jeloModels = ['decisionTreeRegressor', 'ridge', 'lasso', 'elasticNet', 'bayesianRidge']
        selectedJeloModel = st.selectbox('Select Jelo Model', jeloModels, index=None)
        jelo = Jelo()
        jeloMetrics = jelo.fitModel(selectedJeloModel, joinedData) if selectedJeloModel else None
        st.success(f"{selectedJeloModel} model selected successfully.") if selectedJeloModel else None

        st.write('Jelo Model Performance Metrics:')
        st.dataframe(pd.DataFrame(jeloMetrics) if jeloMetrics else pd.DataFrame(self.metrics))

        st.header('Vehicle and Real/Predict Weights')
        vehicleWeights = dict()
        for vehicle, column in zip(['car', 'bus', 'truck', 'bicycle', 'motorcycle'], st.columns(5)):
            with column:
                vehicleWeights[vehicle] = st.number_input(f"{vehicle}", min_value=1.0, step=0.5)


        realWeight = st.slider('Real Weight', min_value=0.0, max_value=1.0, step=0.1)
        predictWeight = round(1.0 - realWeight, 1)
        st.success(f"Real Weghit: {realWeight}, Prediction Weight: {predictWeight} ") if realWeight else None

        st.header('Real-Time YOLOv8 Detection')
        yoloResults, haloResults, jeloResults, weightedYoloResults, weightedHaloResults, combinedResults = None, None, None, None, None, None
        if st.button('Start Analysis'):
            caps = {side: [cv2.VideoCapture(video)] for side, video in videoUploads.items() if video is not None}
            [caps[side].append(int(caps[side][0].get(cv2.CAP_PROP_FRAME_COUNT))) for side in caps.keys()]
            frameCounts = {side: 0 for side in caps.keys()}
            stframes = {side: st.empty() for side in videoUploads.keys()}
            
            
            while all(cap.isOpened() for (cap, _) in caps.values()) and all(frameCounts[side] < int(frames) for side, (_, frames) in caps.items()):
                yoloResults = copy.deepcopy(self.vehicles)
                yoloResults.pop('total')
                yoloResults.update(copy.deepcopy(self.emergency))
                yoloResults.update({'total': []})

                haloResults = copy.deepcopy(self.vehicles)
                jeloResults = copy.deepcopy(self.lights)

                weightedYoloResults = copy.deepcopy(self.vehicles)
                weightedHaloResults = copy.deepcopy(self.vehicles)
                combinedResults = copy.deepcopy(self.vehicles)

                for side, (cap, _) in caps.items():
                    if not cap.isOpened():
                        continue

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frameCounts[side]) 
                    ret, frame = cap.read()

                    if not ret:
                        continue

                    frame = cv2.resize(frame, (1020, 500))
                    
                    yoloFrame, yoloCounts = yolo.visualizeTracking(frame)
                    yoloResults = self.update(yoloResults, yoloCounts, date, time, side)

                    weightedYolo = {vehicle: round(realWeight * yoloResults.get(vehicle, [0])[-1] * vehicleWeights.get(vehicle, 1)) for vehicle in vehicleWeights}
                    weightedYolo.update({'total': sum(weightedYolo.values())})
                    weightedYoloResults = self.update(weightedYoloResults, weightedYolo, date, time, side)

                    haloResults = self.update(haloResults, halo.predict(selectedHaloModel, date, time, side), date, time, side)
                    weightedHalo = {vehicle: round(predictWeight * haloResults.get(vehicle, [0])[-1] * vehicleWeights.get(vehicle, 1)) for vehicle in vehicleWeights}
                    weightedHalo.update({'total': sum(weightedHalo.values())})
                    weightedHaloResults = self.update(weightedHaloResults, weightedHalo, date, time, side)

                    combined = {'date': [date], 'time': [time], 'side': [side]}
                    combined.update({k: [weightedYolo.get(k, 0) + weightedHalo.get(k, 0)] for k in vehicleWeights})
                    combined.update({'total': weightedYolo['total'] + weightedHalo['total']})
                    combinedResults = self.update(combinedResults, combined, date, time, side)

                    jeloResults = self.update(jeloResults, jelo.predict(selectedJeloModel, pd.DataFrame(combined)), date, time, side)

                    stframes[side].image(yoloFrame, channels='BGR')

                    detectionId = self.db.insertVehicles(
                        self.db.dbName, 
                        yoloResults['date'][-1], 
                        yoloResults['time'][-1], 
                        yoloResults['side'][-1], 
                        yoloResults['car'][-1], 
                        yoloResults['bus'][-1], 
                        yoloResults['truck'][-1], 
                        yoloResults['motorcycle'][-1], 
                        yoloResults['bicycle'][-1], 
                        yoloResults['total'][-1]
                    )
                    self.db.insertLights(
                        self.db.dbName, 
                        detectionId, 
                        jeloResults['green'][-1]
                    )

                    print('side: ', side, ', frame: ', frameCounts[side], ' processed successfully.')
                    frameCounts[side] += frameRate

            for (cap, _) in caps.values():
                cap.release()

        st.header('Model Results')
        st.write('YOLO Results:')
        st.dataframe(pd.DataFrame(yoloResults if yoloResults else self.vehicles), hide_index=True)

        st.write('Weighted YOLO Results:')
        st.dataframe(pd.DataFrame(weightedYoloResults if weightedYoloResults else self.vehicles), hide_index=True)

        st.write('Halo Results:')
        st.dataframe(pd.DataFrame(haloResults if haloResults else self.vehicles), hide_index=True)

        st.write('Weighted Halo Results:')
        st.dataframe(pd.DataFrame(weightedHaloResults if weightedHaloResults else self.vehicles), hide_index=True)

        st.write('Combined Weighted YOLO and Halo Results:')
        st.dataframe(pd.DataFrame(combinedResults if combinedResults else self.vehicles), hide_index=True)

        st.write('Jelo Results:')
        if jeloResults:
            for side in range(4):
                idxs = [i for i in range(4) if i != side]
                red = sum(jeloResults['green'][i] for i in idxs)
                jeloResults['red'][side] = red
            st.dataframe(pd.DataFrame(jeloResults), hide_index=True)
        else:
            st.dataframe(pd.DataFrame(self.lights), hide_index=True)