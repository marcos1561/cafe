import pandas as pd
import logging
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from collections import namedtuple

from . import reader
from .turns import TurnList

Mappers = namedtuple('Mappers', [
    'turn_to_id', 'id_to_turn', 
    'week_day_to_id', 'id_to_week_day', 
    'person_to_id', 'id_to_person'
])

class Sheets:
    def __init__(self, 
        pref_path, avail_path, target_work_load_path, people_number_path,
        pref_sheet=None, avail_sheet=None, target_work_load_sheet=None,
        people_number_sheet=None, sheet_name=None,
        ):
        '''
        Sheets with data to generate a schedule.

        OBS: Preference and availability are adjusted to availability be a super set
        of preference and people with no preference have their preference the same as the
        availability.

        Parameters
        ----------
        pref_path:
            People preference data path.

        avail_path:
            People availability data path.
        
        target_work_load_path:
            People desired work load data path.
        
        people_number_path:
            Target people number per turn path.
        '''
        if sheet_name is not None:
            pref_sheet = sheet_name
            avail_sheet = sheet_name
            target_work_load_sheet = sheet_name

        self.pref, week_days1, turns1 = reader.read_schedule(pref_path, pref_sheet, return_week_turns=True)
        self.avail, week_days2, turns2 = reader.read_schedule(avail_path, avail_sheet, True)
        self.target_work_load = reader.read_target_work_load(target_work_load_path, target_work_load_sheet)
        self.people_number = reader.read_people_number(people_number_path, people_number_sheet)

        if week_days1 != week_days2:
            raise Exception((
                "Week days is not the same!\n"
                f"pref week days: {week_days1}"
                f"avail week days: {week_days2}"
            ))
        if turns1 != turns2:
            raise Exception((
                "Turns is not the same!\n"
                f"pref turns: {turns1}"
                f"avail turns: {turns2}"
            ))

        self.adjust_pref_avail()

        self.week_days = week_days1
        self.turns = TurnList(turns1)
        self.people = list(self.avail.keys())
        self.mappers = self.create_mappers()
        
        # If a person does not have a preference, is assumed 
        # that his preference is the same as his availability.
        for p in self.people:
            if p not in self.pref:
                self.pref[p] = self.avail[p]

    def adjust_pref_avail(self):
        for p, day_turn in self.pref.items():
            if p not in self.avail.keys():
                self.avail[p] = self.pref[p]
                continue

            for day, turns in day_turn.items():
                for t in turns:
                    if t not in self.avail[p][day]:
                        self.avail[p][day].add(t)
        
    def create_mappers(self):
        week_days, turns, people = self.week_days, self.turns, self.people

        turn_str_list = [str(t) for t in turns.turns]
        turn_to_id = dict(zip(turn_str_list, range(len(turns))))
        id_to_turn = {v: k for k, v in turn_to_id.items()}

        week_day_to_id = dict(zip(week_days, range(len(week_days))))
        id_to_week_day = {v: k for k, v in week_day_to_id.items()}

        person_to_id = dict(zip(people, range(len(people))))
        id_to_person = {v: k for k, v in person_to_id.items()}

        return Mappers(
            turn_to_id=turn_to_id, id_to_turn=id_to_turn,
            week_day_to_id=week_day_to_id, id_to_week_day=id_to_week_day,
            person_to_id=person_to_id, id_to_person=id_to_person
        )

    def save_pref(self, path):
        self.save_sheet_turns(self.pref, path)
    
    def save_avail(self, path):
        self.save_sheet_turns(self.avail, path)

    def save_sheet_turns(self, sheet: dict, path):
        data = np.full((len(self.turns), len(self.week_days)), "", dtype=object)

        for person, week_turns in sheet.items():
            for day, day_turns in week_turns.items():
                for turn in day_turns:
                    col_id = self.week_days.index(day)
                    row_id = self.turns.turns_str.index(turn)
                    data[row_id, col_id] += f"{person}, " 

        df = pd.DataFrame(data, columns=self.week_days, index=self.turns.turns_str)
        df.to_csv(path)
        
class SchedulerBase(ABC):
    def __init__(self, sheets: Sheets, turns_weights: dict = None):
        self.sheets = sheets
        self.schedule: dict = None
        
        if turns_weights is None:
            turns_weights = {}
        self.turns_weights = turns_weights

    @abstractmethod
    def generate(self):
        pass
    
    def save(self, path: Path, save_pref_avail=True):
        "Save schedule generated at `path` as a .csv"
        week_days = self.sheets.week_days
        turns = self.sheets.turns

        data = []
        for h in turns.turns_str:
            row = h.split("-")
            for d in week_days:
                row.append(", ".join(self.schedule[d][h]))
            data.append(row)

        df = pd.DataFrame(data, columns=["Início", "Fim"] + week_days)
        df.to_csv(path, index=False) 

        if save_pref_avail:
            self.sheets.save_avail(Path(path).parent / "disponibilidade_cafe.csv")
            self.sheets.save_pref(Path(path).parent / "preferencia_cafe.csv")

        import os
        absolute_path = os.path.abspath(path)
        logging.info(f"Escala salva em: {absolute_path}")

    def save_work_load(self, path, target_work_load=None):
        "Calculates total work load per person and save it at `path` as a .csv"
        people = list(self.sheets.avail.keys())
        people_work_num = dict(zip(people, np.zeros(len(people), dtype=float)))
        
        for d, day_sched in self.schedule.items():
            for t, turn_people in day_sched.items():
                for p in turn_people:
                    people_work_num[p] += self.turns_weights.get((d, str(t)), 1)
        
        total_work = []
        for p in people:
            p_work = people_work_num[p]
            if p not in target_work_load:
                w_rel = None
            else:
                w_rel = p_work / target_work_load[p]
            
            total_work.append([p, p_work/2, target_work_load[p]/2, w_rel])

        df = pd.DataFrame(total_work, columns=["Pessoa", "Carga Ganha (Horas)", "Carga Solicitada (Horas)", "Carga Relativa"])
        df.to_csv(path, index=False)

