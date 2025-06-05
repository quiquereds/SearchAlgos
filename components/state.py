class State:
    """
    Clase que representa un estado en un problema de búsqueda. Los estados son utilizados para
    representar los nodos en el grafo de búsqueda. Cada estado tiene un nombre único que lo identifica.
    
    La clase proporciona métodos para comparar estados y obtener su representación
    en forma de cadena.
    """
    def __init__(self, name: str):
        """
        Inicializa el estado con un nombre único.
        Args:
            name (str): Nombre del estado.
        """
        self.name = name
        
    def __eq__(self,other):
        """
        Compara dos estados para verificar si son iguales.
        Args:
            other (State): Otro estado a comparar.
        Returns:
            bool: True si los estados son iguales, False en caso contrario.
        """
        return isinstance(other, State) and self.name == other.name
    
    def __hash__(self):
        """
        Devuelve el hash del estado basado en su nombre.
        Returns:
            int: Hash del estado.
        """
        return hash(self.name)
    
    def __str__(self):
        """
        Devuelve una representación en forma de cadena del estado.
        Returns:
            str: Nombre del estado.
        """
        return self.name