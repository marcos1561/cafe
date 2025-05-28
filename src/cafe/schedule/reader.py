import pandas as pd
import datetime
import numpy as np
from .turns import Turn

# from .constants import week_days

def read_schedule(path, sheet=None, return_week_turns=False):
    '''
    Read schedule from .ods file at `path` in the
    sheet `sheet`.

    Return:
    -------
    schedule:
        Schedule from file in the fallowing form:

        schedule[person name][day] = set with hours this person have selected.
    '''
    # Read the .ods file
    data = pd.read_excel(path, sheet_name=sheet, engine='odf')

    if sheet is None:
        data = list(data.values())[0]

    schedule = {}
    week_days = list(data.columns[2:])
    turns = []

    def hour_to_str(hour):
        if isinstance(hour, datetime.time):
            return f"{hour.hour:02}:{hour.minute:02}"
        elif isinstance(hour, str):
            return ":".join(hour.split(":")[:2])
        else:
            return hour

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
    
    if return_week_turns:
        return schedule, week_days, turns
    else:
        return schedule

def read_target_work_load(path, sheet=None):
    data = pd.read_excel(path, sheet_name=sheet, engine='odf')

    if sheet is None:
        data = list(data.values())[0]

    return dict(zip(data.iloc[:, 0], data.iloc[:, 1] * 2))

def read_target_work_load(path, sheet_name=None):
    data = pd.read_excel(path, sheet_name=sheet_name, engine='odf')

    if sheet_name is None:
        data = list(data.values())[0]

    return dict(zip(data.iloc[:, 0], data.iloc[:, 1] * 2))

def read_people_number(path, sheet_name=None):
    data = pd.read_excel(path, sheet_name=sheet_name, engine='odf')

    if sheet_name is None:
        data = list(data.values())[0]

    data.index = [
        str(Turn(f"{row['Início']}-{row['Fim']}"))
        for _, row in data.iterrows()
    ]

    data.drop(columns=["Início", "Fim"], inplace=True)

    return data

if __name__ == "__main__":
    read_schedule("preferencia.ods")