# cafe
Utilidades para o Café da Física.

# Instalação
Instale o pacote do Café da Física com o seguinte comando
```bash
pip install "git+https://github.com/marcos1561/cafe.git/#egg=cafe"
```
e atualize o mesmo com
```bash
pip install --upgrade "git+https://github.com/marcos1561/cafe.git/#egg=cafe"
```

# Escala
O sub-pacote `cafe.schedule` tem como função gerar a escala semanal do café. Exemplo:
```python
from cafe.schedule import Scheduler, TurnList

# Cria o objeto gerador de escala
# informando qual turno o café abre e fecha
sched = Scheduler(
    open_turn="07:30-08:00",
    close_turn="15:30-16:00",
)

# Exige ter 3 pessoas de 07:00 até 11:00 e
# de 12:30 até 15:30
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

# Gera a escala utilizando as preferências
# e disponibilidades das pessoas
sched.generate(
    preference_path="preferencia.ods",
    availability_path="possibilidade.ods",
    sheet_name="sheet_name",
)

# Mostra onde a disponibilidade foi violada
print("Violações da disponibilidade:")
for people, turns in sched.problems.availability.items():
    print(people, turns)

# Salva a escala como .csv
sched.save("escala.csv")

# Calcula e salva a carga horária semanal
# da escala gerada
sched.calc_work_load("carga.csv")
```
