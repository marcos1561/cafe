# cafe
Utilidades para o Café da Física.

# Instalação
Instale o pacote do Café da Física com o seguinte comando
```bash
pip install -e "git+https://github.com/marcos1561/cafe.git/#egg=cafe"
```

# Escala
O sub-pacote `cafe.schedule` tem como função gerar a escala semanal do café. Exemplo:
```python
from cafe import schedule

# Cria o objeto gerador de escala
# informando qual turno o café abre e fecha
sched = schedule.Scheduler(
    open_turn="07:30-08:00",
    close_turn="15:30-16:00",
)

# Específica quais são os turnos de pico
# exigindo que três pessoas estejam neles.
sched.add_fix_people_turns(
    hours={
        "08:00-08:30", 
        "10:00-10:30", 
        "13:00-13:30", 
        "15:00-15:30",
    },
    people_number=3,
)

# Gera a escala utilizando as preferências
# e disponibilidades das pessoas
sched.generate(
    preference_path="preferencia.ods",
    availability_path="possibilidade.ods",
    sheet_name="sheet_name",
)

# Salva a escala como .csv
sched.save("escala.csv")

# Calcula e salva a carga horários semanal
# da escala gerada
sched.calc_work_load("carga.csv")
```