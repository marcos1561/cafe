import pandas as pd
import datetime
from pathlib import Path

from .turns import Turn

class FileInfo:
    pass

class CsvInfo(FileInfo):
    def __init__(self, path):
        self.path = path

class OdsInfo(FileInfo):
    def __init__(self, path, sheet_name=None):
        self.path = path
        self.sheet_name = sheet_name

def to_path_info(path):
    if not isinstance(path, (str, Path)):
        return path

    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".csv":
        return CsvInfo(path)
    elif extension == ".ods":
        return OdsInfo(path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

def hour_to_str(hour):
    if isinstance(hour, datetime.time):
        return f"{hour.hour:02}:{hour.minute:02}"
    elif isinstance(hour, str):
        return ":".join(hour.split(":")[:2])
    else:
        return hour

def dataframe_to_strange_format(data: pd.DataFrame, return_week_days_and_turns=True):
    "Converts schedule dataframe to a strange format I'm using for unknown reasons."
    schedule = {}
    turns = []
    week_days = list(data.columns[2:])

    # Iterate over data rows
    for _, row in data.iterrows():
        turn = f"{hour_to_str(row['Início'])}-{hour_to_str(row['Fim'])}"
        turns.append(turn)

        for d in week_days:
            names = row[d]
            if type(names) is not str:
                names = ""

            for name in names.split(","):
                name = name.strip()
                if name == "":
                    continue

                if name not in schedule:
                    schedule[name] = {d: set() for d in week_days}

                schedule[name][d].add(turn)
    
    if return_week_days_and_turns:
        return schedule, week_days, turns
    else:
        return schedule

class ScheduleReader:
    @staticmethod
    def read_ods(info: OdsInfo, return_week_days_and_turns=False):
        '''
        Read schedule from .ods file at `path` in the
        sheet `sheet`.

        Return:
        -------
        schedule:
            Schedule from file in the fallowing form:

            schedule[person name][day] = set with hours this person have selected.
        '''
        path, sheet_name = info.path, info.sheet_name
        # Read the .ods file
        data = pd.read_excel(path, sheet_name=sheet_name, engine='odf')

        if sheet_name is None:
            data = list(data.values())[0]

        schedule = {}
        week_days = list(data.columns[2:])
        turns = []

        schedule, week_days, turns = dataframe_to_strange_format(data)
        # def hour_to_str(hour):
        #     if isinstance(hour, datetime.time):
        #         return f"{hour.hour:02}:{hour.minute:02}"
        #     elif isinstance(hour, str):
        #         return ":".join(hour.split(":")[:2])
        #     else:
        #         return hour

        # # Iterate over data rows
        # for _, row in data.iterrows():
        #     turn = f"{hour_to_str(row['Início'])}-{hour_to_str(row['Fim'])}"
        #     turns.append(turn)

        #     for d in week_days:
        #         names = row[d]
        #         if type(names) is not str:
        #             names = ""

        #         for name in names.split(","):
        #             name = name.strip()
        #             if name == "":
        #                 continue

        #             if name not in schedule:
        #                 schedule[name] = {d: set() for d in week_days}

        #             schedule[name][d].add(turn)
        
        if return_week_days_and_turns:
            return schedule, week_days, turns
        else:
            return schedule

    @staticmethod
    def read_csv(info: CsvInfo, return_week_days_and_turns=False):
        path = info.path
        return dataframe_to_strange_format(
            pd.read_csv(path),
            return_week_days_and_turns=return_week_days_and_turns,
        )

    @staticmethod
    def read(file_info: FileInfo, return_week_days_and_turns=False):
        file_info = to_path_info(file_info)
        if isinstance(file_info, CsvInfo):
            func = ScheduleReader.read_csv
        if isinstance(file_info, OdsInfo):
            func = ScheduleReader.read_ods

        return func(file_info, return_week_days_and_turns=return_week_days_and_turns)

class TargetWorkLoadReader:
    @staticmethod
    def read_ods(info: OdsInfo):
        path, sheet_name = info.path, info.sheet_name

        data = pd.read_excel(path, sheet_name=sheet_name, engine='odf')

        if sheet_name is None:
            data = list(data.values())[0]

        return dict(zip(data.iloc[:, 0], data.iloc[:, 1] * 2))

    @staticmethod
    def read_csv(info: CsvInfo):
        path = info.path
        data = pd.read_csv(path, delimiter=",")

        return dict(zip(data.iloc[:, 0], data.iloc[:, 1] * 2))

    @staticmethod
    def read(file_info: FileInfo):
        file_info = to_path_info(file_info)
        if isinstance(file_info, CsvInfo):
            func = TargetWorkLoadReader.read_csv
        if isinstance(file_info, OdsInfo):
            func = TargetWorkLoadReader.read_ods

        return func(file_info)

class TurnCapacityReader:
    @staticmethod
    def read_ods(info: OdsInfo):
        path, sheet_name = info.path, info.sheet_name
        data = pd.read_excel(path, sheet_name=sheet_name, engine='odf')

        if sheet_name is None:
            data = list(data.values())[0]

        data.index = [
            str(Turn(f"{row['Início']}-{row['Fim']}"))
            for _, row in data.iterrows()
        ]

        data.drop(columns=["Início", "Fim"], inplace=True)

        return data

    @staticmethod
    def read(file_info: FileInfo):
        file_info = to_path_info(file_info)
        if isinstance(file_info, OdsInfo):
            func = TurnCapacityReader.read_ods

        return func(file_info)


if __name__ == "__main__":
    pass
    # read_schedule_ods("preferencia.ods")
    # google_reader = GoogleReader()