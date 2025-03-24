from datetime import datetime, timedelta
from datetime import time as TimeType

class Turn:
    def __init__(self, turn: str, duration: timedelta=None):
        '''
        A work turn initializing at `init_hour`.

        Parameter
        ---------
        turn:
            str:
                Turn in the format "HH:MM-HH:MM"
            
            datetime.Time
                start turn time, its duration is specified
                in `duration`.

        duration:
            Turn duration, only applicable if turn is
            datetime.Time.
        '''
        if isinstance(turn, str):
            i, f = turn.split("-")

            self.init = datetime.strptime(i, "%H:%M").time()
            self.end = datetime.strptime(f, "%H:%M").time()
        elif isinstance(turn, TimeType):
            self.init = turn
            self.end = (datetime.combine(datetime.today(), self.init) + duration).time()
        elif isinstance(turn, Turn):
            self.init = turn.init
            self.end = turn.end
            self.duration = turn.duration
            return
        else:
            raise ValueError(f"`turn` should be str, time or Turn, instead it is {type(turn)}.")
        
        self.duration = datetime.combine(datetime.today(), self.end) - datetime.combine(datetime.today(), self.init)

    def __str__(self):
        init = f"{self.init.hour:02}:{self.init.minute:02}"
        end = f"{self.end.hour:02}:{self.end.minute:02}"
        return f"{init}-{end}"

    def __eq__(self, other):
        if isinstance(other, Turn):
            return self.init == other.init and self.end == other.end
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def next(self):
        return Turn(self.end, duration=self.duration)


class TurnList:
    def __init__(self, turns: list[Turn]):
        "List of turns."
        self.turns = [Turn(t) for t in turns]
        self.turns_str = [t.__str__() for t in self.turns]

    @classmethod
    def from_start_end(Cls, start_turn: Turn, end_turn: Turn):
        "Creates list of all turns between `start_turn` and `end_turn`."
        start_turn = Turn(start_turn)
        end_turn = Turn(end_turn)

        turns = [start_turn]
        current_turn = start_turn
        while current_turn != end_turn:
            current_turn = current_turn.next()
            turns.append(current_turn)

        return Cls(turns)
    
    def __add__(self, other):
        if isinstance(other, TurnList):
            combined_turns = self.turns + other.turns
            return TurnList(combined_turns)
        else:
            raise ValueError(f"Cannot add TurnList with {type(other)}")

if __name__ == "__main__":
    a = TurnList.from_start_end("07:30-08:00", "12:00-12:30")
    b = TurnList.from_start_end("14:00-14:30", "15:30-16:00")
    c = a + b
    print(c.turns_str)
    # print(a.turns[0].duration.)