from typing import Dict, Callable, Any, Optional

class SimpleStateMachine:
    """
    A simple state machine that automatically assigns indices based on addition order.
    States are executed in the order they are added.
    """
    
    def __init__(self, max_states: int = -1):
        """
        Initialize the state machine
        
        Args:
            max_states (int): Maximum number of states, -1 means unlimited
        """
        self.state_idx = 0
        self.new_state = True
        self.max_state_cnt = max_states
        self.state_names: Dict[int, str] = {}  # Store state names by index
        self.state_callbacks: Dict[int, Callable] = {}  # Store state callback functions by index
        self.next_index = 0  # Auto-incrementing index for new states
        
    def next(self) -> bool:
        """
        Switch to the next state
        
        Returns:
            bool: True if successfully switched to next state, False otherwise
        """
        if self.max_state_cnt == -1 or self.state_idx < self.max_state_cnt:
            self.new_state = True
            self.state_idx += 1
            return True
        return False

    def trigger(self) -> bool:
        """
        Check if this is a new state
        
        Returns:
            bool: True if this is a new state, False otherwise
        """
        if self.new_state:
            self.new_state = False
            return True
        return False
    
    def reset(self):
        """Reset the state machine to initial state"""
        self.state_idx = 0
        self.new_state = True
        
    def add_state(self, name: str, callback: Optional[Callable] = None) -> int:
        """
        Add a state with its callback function. Index is automatically assigned.
        
        Args:
            name (str): State name
            callback (callable, optional): State callback function
            
        Returns:
            int: The assigned index for this state
        """
        state_idx = self.next_index
        self.state_names[state_idx] = name
        if callback is not None:
            self.state_callbacks[state_idx] = callback
        self.next_index += 1
        return state_idx
        
    def get_state_name(self) -> Optional[str]:
        """
        Get the name of current state
        
        Returns:
            str: Name of current state, None if not defined
        """
        return self.state_names.get(self.state_idx)
        
    def execute_current_state(self, *args, **kwargs) -> Any:
        """
        Execute the callback function of current state
        
        Args:
            *args: Arguments passed to the callback function
            **kwargs: Keyword arguments passed to the callback function
            
        Returns:
            Any: Return value of the callback function, None if no callback function defined
        """
        callback = self.state_callbacks.get(self.state_idx)
        if callback is not None:
            return callback(*args, **kwargs)
        return None
    
    def set_state(self, state_idx: int):
        """
        Set the current state to a specific index
        
        Args:
            state_idx (int): Target state index
        """
        if state_idx in self.state_names:
            self.state_idx = state_idx
            self.new_state = True