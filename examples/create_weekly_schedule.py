from cafe.schedule import Scheduler, TurnList

sched = Scheduler(
    open_turn="07:30-08:00",
    close_turn="16:00-16:30",
)

sched.add_fix_people_turns(
    turns=TurnList.from_start_end(
        start_turn="07:00-07:30",
        end_turn="10:30-11:00", 
    ) + TurnList.from_start_end(
        start_turn="12:30-13:00",
        end_turn="15:00-15:30", 
    ),
    people_number=3,
)

sched.generate(
    preference_path="preferencia.ods",
    availability_path="possibilidade.ods",
    sheet_name="Sheet1",
)

print("Violações da disponibilidade:")
for p, turns in sched.problems.availability.items():
    print(p, turns)

sched.save("escala.csv")
sched.calc_work_load("carga.csv")