import pandas as pd
from .constants import week_days

def read_schedule(path, sheet):
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

    schedule = {}

    # Iterate over data rows
    for _, row in data.iterrows():
        hour = f"{row['Início']}-{row['Fim']}"
        
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

                schedule[name][d].add(hour)
    
    return schedule

if __name__ == "__main__":
    read_schedule("preferencia.ods")