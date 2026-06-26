from RouteData import RouteData
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from scipy.spatial.distance import cdist
from math import floor
import matplotlib.pyplot as plt
from random import random

"""
NOTE:
Given sample data, no solution found >4 vehicles
198 (total demand) / 67 (vehicle max) = >3
But demand per stop point can't be solved with only 3 vans at 67 cap.
"""


class CVRP:
    def __init__(self, coordinates, parcel_demands, vehicles = 4,
                 fuel_cost_gbr_per_m = 0.000055, shift_cost_gbr = 97.44):
        self._parcel_demands = parcel_demands
        self._depot = 0 # The depot MUST be the first point in the coordinates list
        self._vehicles = vehicles
        self._fuel_cost = fuel_cost_gbr_per_m
        self._shift_cost = shift_cost_gbr
        self._coordinates = coordinates
        self._solution = None

        # The distance matrix is a 2D list where the element at [i][j]
        # represents the distance from stop point i to stop point j.
        self._distance_matrix = self.get_distance_matrix(coordinates)

        # The RoutingIndexManager maps the node indices (stop point indices)
        # to the internal indices used by the OR-Tools solver.
        self._manager = pywrapcp.RoutingIndexManager(len(
            self._distance_matrix), vehicles, self._depot)

        # The RoutingModel is the solver that will find the optimal routes
        # given the distance matrix and constraints.
        self._routing = pywrapcp.RoutingModel(self._manager)

    @staticmethod
    def get_binding_capacity(capacities):
        """Returns binding capacity (capacity that limits parcels to the
        lowest amount) for vehicles"""

        # Validates that the capacities dictionary contains required keys.
        if ("vehicle_weight" not in capacities or "vehicle_volume" not in
                capacities or "parcel_weight" not in capacities or "parcel_volume" not in capacities):
            raise ValueError("Capacities dictionary must contain "
                             "vehicle_weight, vehicle_volume, parcel_weight, and parcel_volume.")

        maximum_parcels_by_weight = capacities["vehicle_weight"] / capacities["parcel_weight"]
        maximum_parcels_by_volume = capacities["vehicle_volume"] / capacities["parcel_volume"]

        binding_capacity = floor(min(maximum_parcels_by_weight,
                           maximum_parcels_by_volume))
        # Ensures binding capacity is at least 1 to avoid zero capacity.
        return max(1, binding_capacity)

    @staticmethod
    def get_distance_matrix(coordinates):
        """Returns the Manhattan distance matrix for the given coordinates."""

        # Cdist calculates distance between every pair of stop points (no.
        # of points * no of points matrix)
        # Cityblock gets Manhattan distances (more realistic for roads).
        # Convert to int values and list for OR-Tools compatibility (cdist
        # returns NumPy array of floats).
        return cdist(coordinates, coordinates, metric="cityblock").astype(
            int).tolist()

    def solve(self, capacities):
        """Builds the model, applies capacity, cost, and demand callbacks,
        configures the search strategy, runs the solver, and prints the
        solution if one exists."""

        self.__configure_callbacks(capacities)
        search_params = self.__configure_search_params()
        self._solution = self._routing.SolveWithParameters(search_params)
        if self._solution:
            self.print_solution()
        else:
            print("No solution found!")

    def print_solution(self):
        """Prints details of solution: routes, loads, di
        stances,
        fuel costs, and total operational cost."""

        (route_desc, total_distance, total_fuel_cost, total_load,
         vehicles_used) = (self.__format_solution_details())

        # Print summary per route
        for i in route_desc:
            print(i)

        shift_cost = vehicles_used * self._shift_cost
        total_cost = total_fuel_cost + shift_cost

        # Print overall solution summary
        print(f"Total distance travelled: "
              f"{self.__format_int(total_distance)}m")
        print(f"Vehicles used: {vehicles_used}")
        print(f"Total shift cost: £{shift_cost}")
        print(f"Total load of all vehicles: {total_load}")
        print(f"Combined objective (fuel + shift cost): "
              f"{self.__format_int(self._solution.ObjectiveValue() + shift_cost)}")
        print(f"Total operational cost: £{total_cost}")

    def show_route_visualisations(self):
        """Display matplotlib visualisations of solution"""

        if not self._solution:
            print("No solution found!")
            return

        self.plot_routes()
        self.plot_route_distances()

    def plot_routes(self):
        """Plot all routes on a graph"""

        plt.figure(figsize=(10, 8), facecolor="orange")
        # Plot stop points
        x_coords = [coord[0] for coord in self._coordinates]
        y_coords = [coord[1] for coord in self._coordinates]
        plt.scatter(x_coords, y_coords, s=20, label="Stop point")

        # Highlight depot with red square
        depot_x, depot_y = self._coordinates[0]
        plt.scatter([depot_x], [depot_y], c="red", s=120, marker="s",
                    label="Depot")

        # Assign random colours for each route
        for route in self.__get_solution_routes_iter():
            colour = (random(), random(), random())
            route_x = []
            route_y = []

            # Store x, y coords for route stop points
            for node in route.node_sequence:
                x, y = self._coordinates[node]
                route_x.append(x)
                route_y.append(y)

            plt.plot(route_x, route_y, "-o", color=colour,
                     label=f"Vehicle {route.vehicle_id + 1}")

        plt.title("Vehicle Routes")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_route_distances(self):
        """Plot bar chart showing total distance traversed by each used
        vehicle."""

        distances = {}
        # Get distances traversed by each vehicle
        for route in self.__get_solution_routes_iter():
            distances[f"V{route.vehicle_id + 1}"] = route.distance

        # Plot
        plt.figure(figsize=(10, 6), facecolor="orange")
        plt.gca().set_facecolor("black")
        plt.bar(distances.keys(), distances.values(), color="red")
        plt.title("Route Distances per Vehicle")
        plt.xlabel("Vehicle")
        plt.ylabel("Distance (m)")
        plt.grid(False)
        plt.show()

    def __configure_callbacks(self, capacities):
        """Configures distance, demand, and fuel cost callbacks.
        Callbacks are functions that the solver calls to get the
        distance/demand/fuel cost between nodes."""

        # Tells the solver how to compute the distance between two nodes.
        self._routing.RegisterTransitCallback(
            self.__distance_callback)

        # Tells the solver how to compute the fuel cost between two nodes.
        fuel_callback_index = self._routing.RegisterTransitCallback(
            self.__fuel_cost_callback
        )
        # Sets the fuel cost as the arc cost (objective) for all vehicles.
        # This is calculated with distance in the callback method.
        self._routing.SetArcCostEvaluatorOfAllVehicles(fuel_callback_index)

        # Tells the solver how to compute the demand of each node.
        demand_callback_index = self._routing.RegisterUnaryTransitCallback(
            self.__demand_callback)

        # Adds a capacity dimension to the solver, which enforces that
        # the total demand on each route does not exceed the vehicle capacity.
        vehicle_capacity = self.get_binding_capacity(capacities)
        self._routing.AddDimension(
            demand_callback_index, # Method to get the demand of each node
            0,  # Slack for capacity (0 = no overflow allowed)
            vehicle_capacity,  # Vehicle maximum capacities
            True,  # Start each vehicle at zero load
            "Capacity", # Name of the dimension
        )
        # Sets a fixed cost for using each vehicle, which may encourage the
        # solver to use fewer vehicles if possible.
        self._routing.SetFixedCostOfAllVehicles(round(self._shift_cost))

    @staticmethod
    def __configure_search_params():
        """Configures the search settings for the solver."""

        # Sets up default configuration
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        # Sets first solution strategy which is the heuristic used by solver
        # to choose order of node traversal.
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy
            .PARALLEL_CHEAPEST_INSERTION
        )

        # Sets local search metaheuristic to tell the solver how
        # to improve the solution. GLS penalises costly routes.
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )

        # Sets time limit for solve to find solution; if no solution can be
        # found, solver will return None.
        search_parameters.time_limit.FromSeconds(10)
        return search_parameters

    def __get_route_data(self, vehicle_id):
        if not self.__is_vehicle_used(vehicle_id):
            return None

        # Start stop point for vehicle (depot)
        index = self._routing.Start(vehicle_id)
        node_sequence = []
        route_distance = 0
        route_fuel_cost = 0
        route_load = 0

        # Until end point for vehicle (depot)
        while not self._routing.IsEnd(index):
            # Converts internal index to stop point node
            node = self._manager.IndexToNode(index)
            node_sequence.append(node) # Track route
            # Track parcels per stop point
            route_load += self._parcel_demands[node]

            # Track previous stop point
            previous_index = index
            # Move to next stop point
            index = self._solution.Value(self._routing.NextVar(index))
            next_node = self._manager.IndexToNode(index)

            # Track total route distance through distance between stop points
            route_distance += self._distance_matrix[node][next_node]
            # Track total route fuel cost
            route_fuel_cost += self._routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        # Append final stop point to route (depot)
        node_sequence.append(self._manager.IndexToNode(index))
        # RouteData acts as a record
        return RouteData(
            vehicle_id=vehicle_id,
            node_sequence=node_sequence,
            distance=route_distance,
            fuel_cost=route_fuel_cost,
            load=route_load,
        )

    def __format_route_description(self, route):
        route_plan = f"Route for vehicle {route.vehicle_id + 1}:\n"
        accumulated_load = 0

        # Zip is used to combine iterables; results in tuples of each stop
        # point and its next one
        for current_node, next_node in zip(route.node_sequence,
                                          route.node_sequence[1:]):
            # Add stop point identifier & parcel demand to route plan
            if current_node == 0:
                route_plan += "Depot --"
            else:
                accumulated_load += self._parcel_demands[current_node]
                route_plan += f"P{current_node}: {accumulated_load} parcels --"

            # Add distance in m to next stop point to route plan
            distance_to_next = self._distance_matrix[current_node][next_node]
            route_plan += f"{self.__format_int(distance_to_next)}m--> "

        route_plan += "Depot\n"
        route_plan += (f"Distance of route: "
                       f"{self.__format_int(route.distance)}m\n")
        route_plan += f"Fuel cost of route: {route.fuel_cost}\n"
        route_plan += f"Parcel load of route: {route.load}\n"
        return route_plan

    def __format_solution_details(self):
        """Retrieves components used in solution, allowing them to be
        formatted for display. Returns optimised routes, total_distance,
        total_fuel_cost, total_load, and total vehicles_used as a tuple."""

        total_distance = 0
        total_fuel_cost = 0
        total_load = 0
        vehicles_used = 0
        route_descriptions = []

        # Get all route details and summary from solution routes iterable
        for route in self.__get_solution_routes_iter():
            vehicles_used += 1
            route_descriptions.append(self.__format_route_description(route))
            total_distance += route.distance
            total_fuel_cost += route.fuel_cost
            total_load += route.load

        return (route_descriptions, total_distance, total_fuel_cost,
                total_load, vehicles_used)

    def __get_solution_routes_iter(self):
        """Helper to get route data for each vehicle"""

        for vehicle_id in range(self._vehicles):
            route = self.__get_route_data(vehicle_id)
            if route:
                yield route

    def __is_vehicle_used(self, vehicle):
        """Helper to find if solution used vehicle"""

        return self._routing.IsVehicleUsed(self._solution, vehicle)

    @staticmethod
    def __format_int(n: int) -> str:
        """Helper to add commas for every hundred in an integer"""

        return f"{n:,}"

    def __distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""

        # IndexToNode() converts internal index to stop point
        from_node = self._manager.IndexToNode(from_index)
        to_node = self._manager.IndexToNode(to_index)
        return self._distance_matrix[from_node][to_node]

    def __demand_callback(self, from_index):
        """Returns the demand of the node."""

        from_node = self._manager.IndexToNode(from_index)
        return self._parcel_demands[from_node]

    def __fuel_cost_callback(self, from_index, to_index):
        """Returns the fuel cost between the two nodes."""

        distance = self.__distance_callback(from_index, to_index)
        fuel_cost_per_metre = self._fuel_cost  # £ per metre
        return round(distance * fuel_cost_per_metre)


if __name__ == "__main__":
    stop_point_coordinates = [
        (554099, 191976),
        (550737.8, 198281.9),
        (550125.2, 195507.2),
        (551727.6, 194833.1),
        (548007.6, 194890.4),
        (548207.5, 197457.1),
        (553795.4, 197586.9),
        (553324.3, 193948.7),
        (550536, 187929),
        (551076.7, 193107.2),
        (546698, 196748.4)
    ]
    parcel_demand_per_stop_point = [0, 16, 25, 25, 20, 12, 10, 12, 1, 19, 58]
    capacity_constraints = {
        "vehicle_weight": 780,  # kg
        "vehicle_volume": 4.4,   # m^3
        "parcel_weight": 10,     # kg
        "parcel_volume": 0.065     # m^3
    }

    cvrp = CVRP(stop_point_coordinates, parcel_demand_per_stop_point)
    cvrp.solve(capacity_constraints)
    cvrp.show_route_visualisations()