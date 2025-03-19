from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus
import pandas as pd
from collections import namedtuple

from . import reader
from .constants import *

FixPeopleHours = namedtuple('FixPeopleHours', ['hours', 'people_number'])

class Scheduler:
    def __init__(self, open_turn: str, close_turn: str, desired_work_load: dict=None,
        max_open_per_people=2, max_close_per_people=2, max_load_per_day=None,         
    ):
        '''
        System to generate a schedule.

        OBS: 
            A turn is a string with the fallowing structure: "HH:MM-HH:MM".
            Example: "07:30-08:00" 

        Parameters
        ----------
        open_turn:
            When the cafe opens
        
        close_turn:
            When the cafe closes

        desired_work_load:
            Map between people and its desired total work load in the week in hours.
        
        max_open_per_people:
            Maximum number of times a given person open the cafe
        
        max_close_per_people:
            Maximum number of times a given person close the cafe

        max_load_per_day:
            Maximum number of hours a given person works per day. If None there is no limit.
        '''
        self.open_turn = open_turn
        self.close_turn = close_turn
        
        self.max_open_per_people = max_open_per_people
        self.max_close_per_people = max_close_per_people

        if desired_work_load is None:
            desired_work_load = {}
        self.work_load = desired_work_load

        self.max_load_per_day = max_load_per_day

        self.fix_people_hours: list[FixPeopleHours] = []

        self.x = None
        self.schedule = None

    def add_fix_people_hours(self, hours: str, people_number: int):
        "Add hours where there should be `people_number` people at the cafe."
        self.fix_people_hours.append(
            FixPeopleHours(hours=hours, people_number=people_number)
        )

    def generate(self, preference_path, availability_path, sheet_name):
        preference: dict = reader.read_schedule(preference_path, sheet_name)
        availability: dict = reader.read_schedule(availability_path, sheet_name)

        for p in preference.keys():
            if p not in availability.keys():
                availability[p] = preference[p]

        peoples = list(availability.keys())

        # Creating decision variables x[person][day][hour]
        x = {
            p: {
                d: {h: LpVariable(f"x_{p}_{d}_{h}", cat="Binary") for h in work_turns}
                for d in week_days
            }
            for p in peoples
        }

        # Creating the maximization problem (prioritizing preferences)
        prob = LpProblem("Weekly_Schedule_Generation", LpMaximize)

        # Objective function: Maximize allocations in preferred hours
        prob += lpSum(
            x[p][d][h] for p in preference for d in week_days for h in preference[p][d] if h in x[p][d]
        ), "Maximize Preferred Hours"

        # Constraint: Each shift needs 2 or 3 people
        for fix_people_hours in self.fix_people_hours:
            people_number = fix_people_hours.people_number
            hours = fix_people_hours.hours

            for d in week_days:
                for h in work_turns:
                    if h in hours:
                        prob += lpSum(x[p][d][h] for p in peoples) == people_number, f"Shift_{d}_{h}_{people_number}_People"
                    else:
                        prob += lpSum(x[p][d][h] for p in peoples) <= 2, f"Shift_{d}_{h}_max_2_people"
                        prob += lpSum(x[p][d][h] for p in peoples) >= 1, f"Shift_{d}_{h}_min_1_people"
                

        # Constraint: Ensure that each person is only scheduled when available
        for p in peoples:
            for d in week_days:
                for h in work_turns:
                    if h not in availability[p][d]: 
                        prob += x[p][d][h] == 0, f"Unavailability_{p}_{d}_{h}"


        # Constraint: Ensure that each person works close to the desired weekly workload
        delta_h = 1
        for p in peoples:
            if p not in self.work_load:
                continue
            
            prob += (
                lpSum(x[p][d][h] for d in week_days for h in work_turns) >= self.work_load[p] - 2*delta_h,
                f"Load_Min_{p}"
            )
            prob += (
                lpSum(x[p][d][h] for d in week_days for h in work_turns) <= self.work_load[p] + 2*delta_h,
                f"Load_Max_{p}"
            )


        # Constraint: Distribute who opens and closes the day evenly throughout the week
        for p in peoples:
            prob += lpSum(
                x[p][d][self.open_turn] for d in week_days if self.open_turn in availability[p].get(d, set())
            ) <= self.max_open_per_people, f"Balanced_Weekly_Opening_{p}"
            
            prob += lpSum(
                x[p][d][self.close_turn] for d in week_days if self.close_turn in availability[p].get(d, set())
            ) <= self.max_close_per_people, f"Balanced_Weekly_Closing_{p}"


        # Constraint: Ensure everyone doesn't exceed the maximum work load per day
        if self.max_load_per_day is not None:
            for p in peoples:
                for d in week_days:
                    prob += lpSum(x[p][d][h] for h in work_turns) <= self.max_load_per_day*2, f"Max_Load_{p}_{d}"

        prob.solve()

        # Display solution status
        print(f"Status da solução: {LpStatus[prob.status]}\n")

        # Generate schedule
        schedule = {d: {h: [] for h in work_turns} for d in week_days}
        for p in peoples:
            for d in week_days:
                for h in work_turns:
                    if x[p][d][h].value() == 1:
                        schedule[d][h].append(p)
        
        self.schedule = schedule
        self.x = x

    def show(self):
        "Show schedule generated."
        for d in week_days:
            print(f"\nEscala para {d.capitalize()}:")
            for h in work_turns:
                print(f"{h}: {', '.join(self.schedule[d].get(h, []))}")

    def save(self, path):
        "Save schedule generated at `path` as a .csv"
        data = []
        for h in work_turns:
            row = h.split("-")
            for d in week_days:
                row.append(", ".join(self.schedule[d][h]))
            data.append(row)

        df = pd.DataFrame(data, columns=["Início", "Fim"] + week_days)
        df.to_csv(path, index=False) 

    def calc_work_load(self, path):
        "Calculates total work load per person and save it at `path` as a .csv"
        total_work = []
        for p in self.x.keys():
            p_work = 0
            for d in week_days:
                for h in work_turns:
                    p_work += self.x[p][d][h].value()
            total_work.append([p, p_work / 2])
        
        df = pd.DataFrame(total_work, columns=["Pessoa", "Carga Horária (Horas)"])
        df.to_csv(path, index=False)
