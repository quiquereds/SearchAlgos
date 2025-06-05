class Action:
    """
    Clase que representa una acción en un problema de búsqueda. Las acciones son utilizadas para transitar
    entre estados en el grafo de búsqueda. Cada acción tiene un nombre único que la identifica.
    
    La clase proporciona métodos para comparar acciones y obtener su representación
    en forma de cadena.
    """
    def __init__(self, name: str):
        """
        Inicializa la acción con un nombre único.
        Args:
            name (str): Nombre de la acción.
        """
        self.name = name
        
    def __eq__(self, other):
        """
        Compara dos acciones para verificar si son iguales.
        Args:
            other (Action): Otra acción a comparar.
        Returns:
            bool: True si las acciones son iguales, False en caso contrario.
        """
        return isinstance(other, Action) and self.name == other.name
    
    def __hash__(self):
        """
        Devuelve el hash de la acción basado en su nombre.
        Returns:
            int: Hash de la acción.
        """
        return hash(self.name)
    
    def __str__(self):
        """
        Devuelve una representación en forma de cadena de la acción.
        Returns:
            str: Nombre de la acción.
        """
        return self.name
    
    