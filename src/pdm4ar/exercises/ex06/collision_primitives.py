from pdm4ar.exercises_def.ex06.structures import *
from triangle import triangulate
import triangle as tr
import numpy as np
from typing import Union, Optional


class CollisionPrimitives_SeparateAxis:
    """
    Class for Implementing the Separate Axis Theorem


    A docstring with expected inputs and outputs is provided for each of the functions
    you are to implement. You do not need to adhere to the given variable names, but you should adhere to
    the datatypes of inputs and outputs.

    ## THEOREM
    Let A and B be two disjoint nonempty convex subsets of R^n. Then there exist a nonzero vector v anda  real number c s.t.
    <x,v> >= c and <y,v> <= c. For all x in A and y in B. i.e. the hyperplane <.,v> = c separates A and B.

    If both sets are closed, and at least one of them is compact, then the separation can be strict, that is,
    <x,v> > c_1 and <y,v> < c_2 for some c_1 > c_2


    In this exercise, we will be implementing the Separating Axis Theorem for 2d Primitives.

    """

    # Task 1
    @staticmethod
    def proj_polygon(p: Union[Polygon, Circle, Triangle], ax: Segment) -> Segment:
        """
        Project the Polygon onto the axis, represented as a Segment.
        Inputs:
        Polygon p,
        a candidate axis ax to project onto

        Outputs:
        segment: a (shorter) segment with start and endpoints of where the polygon has been projected to.

        """
        p1 = np.array([ax.p1.x, ax.p1.y])
        p2 = np.array([ax.p2.x, ax.p2.y])
        d = p2 - p1
        d_normalized = d / np.linalg.norm(d)

        if isinstance(p, Polygon):

            v = np.array([[vertex.x, vertex.y] for vertex in p.vertices])
            projections = np.dot(v - p1, d_normalized)
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            min_point = p1 + min_proj * d_normalized
            max_point = p1 + max_proj * d_normalized

            return Segment(Point(min_point[0], min_point[1]), Point(max_point[0], max_point[1]))
        if isinstance(p, Circle):
            center = np.array([p.center.x, p.center.y])
            v = center - p1
            proj_length = np.dot(v, d_normalized)
            proj_center = p1 + proj_length * d_normalized
            proj_start = proj_center - d_normalized * p.radius
            proj_end = proj_center + d_normalized * p.radius
            return Segment(Point(proj_start[0], proj_start[1]), Point(proj_end[0], proj_end[1]))
        if isinstance(p, Triangle):
            pass

    # Task 2.a
    @staticmethod
    def overlap(s1: Segment, s2: Segment) -> bool:
        """
        Check if two segments overlap.
        Inputs:
        s1: a Segment
        s2: a Segment

        Outputs:
        bool: True if segments overlap. False o.w.
        """

        seg1_start, seg1_end = min(s1.p1.x, s1.p2.x), max(s1.p1.x, s1.p2.x)
        seg2_start, seg2_end = min(s2.p1.x, s2.p2.x), max(s2.p1.x, s2.p2.x)
        return (seg1_start <= seg2_end) and (seg2_start <= seg1_end)

    # Task 2.b
    @staticmethod
    def get_axes(p1: Polygon, p2: Polygon) -> list[Segment]:
        """
        Get all Candidate Separating Axes.
        Hint: These are 2D Polygons, recommend searching over axes that are orthogonal to the edges only.
        Rather than returning infinite Segments, return one axis per Edge1-Edge2 pairing.

        Inputs:
        p1, p2: Polygons to obtain separating Axes over.
        Outputs:
        list[Segment]: A list of segments of size N (worldlength) that represent each of the valid separating axes for the two polygons.
        """

        vertices1 = np.array([[v.x, v.y] for v in p1.vertices])
        edges1 = np.diff(np.vstack([vertices1, vertices1[0]]), axis=0)
        normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        # norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
        axes1 = normals1  # / norms1

        vertices2 = np.array([[v.x, v.y] for v in p2.vertices])
        edges2 = np.diff(np.vstack([vertices2, vertices2[0]]), axis=0)
        normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        # norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
        axes2 = normals2  # / norms2

        all_axes = np.vstack([axes1, axes2])
        abs_axes = np.abs(all_axes)
        sum_abs_axes = abs_axes.sum(axis=1)
        min_abs_axis = all_axes[np.argmin(sum_abs_axes)]
        k = min(1 / np.abs(min_abs_axis).min(), 1)
        starts = -all_axes * 20 * k
        ends = all_axes * 20 * k
        axes = []
        for start, end in zip(starts, ends):
            axes.append(Segment(Point(start[0], start[1]), Point(end[0], end[1])))
        # axes = []
        # for ax in np.vstack([axes1, axes2]):
        #     start = Point(-ax[0] * 20, -ax[1] * 20)
        #     end = Point(ax[0] * 20, ax[1] * 20)
        #     axes.append(Segment(start, end))

        return axes

    # Task 2.c
    @staticmethod
    def separating_axis_thm(
        p1: Polygon,
        p2: Union[Polygon, Circle],
    ) -> tuple[bool, Optional[Segment]]:
        """
        Get Candidate Separating Axes.
        Once obtained, loop over the Axes, project the polygons onto each acis and check overlap of the projected segments.
        If an axis with a non-overlapping projection is found, we can terminate early. Conclusion: The polygons do not collide.

        IMPORTANT
        This Method Evaluates task 2 and Task 3.
        Task 2 checks the separate axis theorem for two polygons.
        Task 3 checks the separate axis theorem for a circle and a polygon
        We have provided a skeleton on this method to distinguish the two test cases, feel free to use any helper methods above, but your output must come
        from  separating_axis_thm().

        Hint: You can use previously implemented methods, such as overlap() and get_axes()

        Inputs:
        p1, p2: Candidate Polygons
        Outputs:
        Output as a tuple
        bool: True if Polygons Collide. False o.w.
        Segment: An Optional argument that allows you to visualize the axis you are projecting onto.
        """

        if isinstance(p2, Polygon):  # Task 2c

            axes = CollisionPrimitives_SeparateAxis.get_axes(p1, p2)
            for axis in axes:
                # Project both polygons onto the axis
                proj1 = CollisionPrimitives_SeparateAxis.proj_polygon(p1, axis)
                proj2 = CollisionPrimitives_SeparateAxis.proj_polygon(p2, axis)

                # Check for overlap
                if not CollisionPrimitives_SeparateAxis.overlap(proj1, proj2):
                    # If projections do not overlap, return False and the axis
                    return False, axis

            # If all projections overlap, the polygons collide
            return True, None

        elif isinstance(p2, Circle):  # Task 3b

            axes = CollisionPrimitives_SeparateAxis.get_axes_cp(p2, p1)
            for axis in axes:
                # Project both polygons onto the axis
                proj1 = CollisionPrimitives_SeparateAxis.proj_polygon(p1, axis)
                proj2 = CollisionPrimitives_SeparateAxis.proj_polygon(p2, axis)

                # Check for overlap
                if not CollisionPrimitives_SeparateAxis.overlap(proj1, proj2):
                    # If projections do not overlap, return False and the axis
                    return False, axis

            # If all projections overlap, the polygons collide
            return True, None

        else:
            print("If we get here we have done a big mistake - TAs")
            return (bool, axis)

    # Task 3
    @staticmethod
    def get_axes_cp(circ: Circle, poly: Polygon) -> list[Segment]:
        """
        Get all Candidate Separating Axes.
        Hint: Notice that the circle is a polygon with infinite number of edges. Fortunately we do not need to check all axes normal to the edges.
        It's sufficient to check the axes normal to the polygon edges plus ONE axis formed by the circle center and the closest vertice of the polygon.

        Inputs:
        circ, poly: Cicle and Polygon to check, respectively.
        Ouputs:
        list[Segment]: A list of segments of size N (worldlength) that represent each of the valid separating axes for the two polygons.
        """
        axes = []

        vertices = np.array([[v.x, v.y] for v in poly.vertices])
        edges1 = np.diff(np.vstack([vertices, vertices[0]]), axis=0)
        normals = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        # norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # axes1 = normals / norms
        axes1 = normals

        center = np.array([circ.center.x, circ.center.y])
        distances = np.linalg.norm(center - vertices, axis=1)
        closest_idx = np.argmin(distances)
        line = center - vertices[closest_idx]
        # extra_ax = [line / np.linalg.norm(line)]
        extra_ax = line

        all_axes = np.vstack([axes1, extra_ax])
        abs_axes = np.abs(all_axes)
        sum_abs_axes = abs_axes.sum(axis=1)
        min_abs_axis = all_axes[np.argmin(sum_abs_axes)]
        k = min(1 / np.abs(min_abs_axis).min(), 1)
        starts = -all_axes * 20 * k
        ends = all_axes * 20 * k

        for start, end in zip(starts, ends):
            axes.append(Segment(Point(start[0], start[1]), Point(end[0], end[1])))
        # for ax in np.vstack([axes1, extra_ax]):
        #     start = Point(-ax[0] * 20, -ax[1] * 20)
        #     end = Point(ax[0] * 20, ax[1] * 20)
        #     axes.append(Segment(start, end))

        return axes


class CollisionPrimitives:
    """
    Class of collision primitives
    """

    NUMBER_OF_SAMPLES = 100

    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        """
        Given function.
        Checks if a circle and a point are in collision.

        Inputs:
        c: Circle primitive
        p: Point primitive

        Outputs:
        bool: True if in Collision, False otherwise
        """
        return (p.x - c.center.x) ** 2 + (p.y - c.center.y) ** 2 < c.radius**2

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        """
        Given function.
        Checks if a Triangle and a Point are in Collision

        Inputs:
        t: Triangle Primitive
        p: Point Primitive

        Outputs:
        bool: True if in Collision, False otherwise.
        """
        area_orig = np.abs((t.v2.x - t.v1.x) * (t.v3.y - t.v1.y) - (t.v3.x - t.v1.x) * (t.v2.y - t.v1.y))

        area1 = np.abs((t.v1.x - p.x) * (t.v2.y - p.y) - (t.v2.x - p.x) * (t.v1.y - p.y))
        area2 = np.abs((t.v2.x - p.x) * (t.v3.y - p.y) - (t.v3.x - p.x) * (t.v2.y - p.y))
        area3 = np.abs((t.v3.x - p.x) * (t.v1.y - p.y) - (t.v1.x - p.x) * (t.v3.y - p.y))

        if np.abs(area1 + area2 + area3 - area_orig) < 1e-3:
            return True

        return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        """
        Given function.

        Input:
        poly: Polygon primitive
        p: Point primitive

        Outputs
        bool: True if in Collision, False otherwise.
        """
        triangulation_result = tr.triangulate(dict(vertices=np.array([[v.x, v.y] for v in poly.vertices])))

        triangles = [
            Triangle(
                Point(triangle[0, 0], triangle[0, 1]),
                Point(triangle[1, 0], triangle[1, 1]),
                Point(triangle[2, 0], triangle[2, 1]),
            )
            for triangle in triangulation_result["vertices"][triangulation_result["triangles"]]
        ]

        for t in triangles:
            if CollisionPrimitives.triangle_point_collision(t, p):
                return True

        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        """
        Given function

        Input:
        c: Circle primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        inside_1 = CollisionPrimitives.circle_point_collision(c, segment.p1)
        inside_2 = CollisionPrimitives.circle_point_collision(c, segment.p2)

        if inside_1 or inside_2:
            return True

        dist_x = segment.p1.x - segment.p2.x
        dist_y = segment.p1.y - segment.p2.y
        segment_len = np.sqrt(dist_x**2 + dist_y**2)

        dot = (
            ((c.center.x - segment.p1.x) * (segment.p2.x - segment.p1.x))
            + ((c.center.y - segment.p1.y) * (segment.p2.y - segment.p1.y))
        ) / pow(segment_len, 2)

        closest_point = Point(
            segment.p1.x + (dot * (segment.p2.x - segment.p1.x)),
            segment.p1.y + (dot * (segment.p2.y - segment.p1.y)),
        )

        # Check whether point is on the segment segment or not
        segment_len_1 = np.sqrt((segment.p1.x - closest_point.x) ** 2 + (segment.p1.y - closest_point.y) ** 2)
        segment_len_2 = np.sqrt((segment.p2.x - closest_point.x) ** 2 + (segment.p2.y - closest_point.y) ** 2)

        if np.abs(segment_len_1 + segment_len_2 - segment_len) > 1e-3:
            return False

        closest_dist = np.sqrt((c.center.x - closest_point.x) ** 2 + (c.center.y - closest_point.y) ** 2)

        if closest_dist < c.radius:
            return True

        return False

    @staticmethod
    def sample_segment(segment: Segment) -> list[Point]:

        x_diff = (segment.p1.x - segment.p2.x) / CollisionPrimitives.NUMBER_OF_SAMPLES
        y_diff = (segment.p1.y - segment.p2.y) / CollisionPrimitives.NUMBER_OF_SAMPLES

        return [
            Point(x_diff * i + segment.p2.x, y_diff * i + segment.p2.y)
            for i in range(CollisionPrimitives.NUMBER_OF_SAMPLES)
        ]

    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        """
        Given function.

        Input:
        t: Triangle Primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:
            if CollisionPrimitives.triangle_point_collision(t, point):
                return True

        return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        """
        Given function.

        Input:
        p: Polygon primitive
        segment: segment primitive

        Outputs:
        bool: True if in collision, False otherwise
        """
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:
            if CollisionPrimitives.polygon_point_collision(p, point):
                return True

        return False

    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:
        """
        Given Function
        Casts a polygon into an AABB, then determines if the bounding box and a segment are in collision

        Inputs:
        p: Polygon primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        aabb = CollisionPrimitives._poly_to_aabb(p)
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:

            if aabb.p_min.x > point.x or aabb.p_min.y > point.y:
                continue

            if aabb.p_max.x < point.x or aabb.p_max.y < point.y:
                continue

            if CollisionPrimitives.polygon_point_collision(p, point):
                return True

        return False

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        """
        Given Function
        Casts a Polygon type into an AABB

        Inputs:
        g: Polygon

        Outputs:
        AABB: Bounding Box for the Polygon.
        """
        x_values = [v.x for v in g.vertices]
        y_values = [v.y for v in g.vertices]

        return AABB(Point(min(x_values), min(y_values)), Point(max(x_values), max(y_values)))
