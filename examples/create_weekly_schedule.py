from cafe.schedule import TurnList, Sheets
from cafe.schedule.annealing import AnnealingSched, SystemParams, TempScalingLaw

# Parâmetros do algoritmo de Annealing
params = SystemParams(
    k_pref=1,
    k_disp=100,
    k_border=10,
    k_fix_people_gt=100,
    k_fix_people_sm=3,
    k_fix_people_sm_peak=2*10,
    k_work_load=0.6,
    k_overflow_work_load=0.6,
    k_lunch=11,
    k_continuos_lunch=11,
    k_no_people=1,
    temp_strat=TempScalingLaw(
        num_steps=100,
        t1=10,
        exponent=0.6,
    )
)

# Criação do escalador especificando horários de pico, hora do almoço e
# peso dos horários.
sched = AnnealingSched(
    params=params,
    init_state="random",
    sheets=Sheets(
        pref_path="preferencia.ods",
        avail_path="possibilidade.ods",
        target_work_load_path="carga_solicitada.ods",
        turn_capacity_path="people_number.ods",
    ),
    peak_turns=TurnList([
        "07:30-08:00", "08:00-08:30", "10:00-10:30", 
        "11:30-12:00", "13:00-13:30", "13:00-13:30", 
        "15:00-15:30",
    ]),
    lunch_turns=TurnList.from_start_end(
        "11:30-12:00", "13:00-13:30",
    ),
    turns_weights={
        ("Segunda", "07:30-08:00"): 2, 
        ("Terça", "07:30-08:00"): 2, 
        ("Quarta", "07:30-08:00"): 2,
        ("Quinta", "07:30-08:00"): 2, 
        ("Sexta", "07:30-08:00"): 2,
        ("Sexta", "16:30-17:00"): 3,
    },
)

# Gera a escala, mostra seus problemas (se houver), salva a carga de trabalho
# da escala gerada e mostra a evolução temporal da energia do algoritmo de Annealing.
sched.generate()
sched.show_problems()
sched.save_work_load("work_load.csv")
sched.save("schedule.csv")
sched.show_energy()