def _infer_reading_order(elements, row_tolerance):
    if len(elements) <= 1:
        return list(elements)

    def xy_cut(elems):
        if len(elems) <= 1:
            return elems
            
        # Try horizontal cut (split top and bottom)
        elems_y = sorted(elems, key=lambda e: e.bbox.y)
        for i in range(1, len(elems_y)):
            max_bottom = max(e.bbox.y2 for e in elems_y[:i])
            # add a small tolerance to gap detection to avoid overlaps breaking cuts
            if elems_y[i].bbox.y > max_bottom - 2.0:
                return xy_cut(elems_y[:i]) + xy_cut(elems_y[i:])
                
        # Try vertical cut (split left and right)
        elems_x = sorted(elems, key=lambda e: e.bbox.x)
        for i in range(1, len(elems_x)):
            max_right = max(e.bbox.x2 for e in elems_x[:i])
            if elems_x[i].bbox.x > max_right - 2.0:
                return xy_cut(elems_x[:i]) + xy_cut(elems_x[i:])
                
        # No clear cut found, fall back
        return _row_sort(elems, row_tolerance)

    def _row_sort(elems, tol):
        sorted_elems = sorted(elems, key=lambda e: (e.bbox.cy, e.bbox.cx))
        rows = []
        current_row = [sorted_elems[0]]
        for elem in sorted_elems[1:]:
            if abs(elem.bbox.cy - current_row[0].bbox.cy) <= tol:
                current_row.append(elem)
            else:
                rows.append(current_row)
                current_row = [elem]
        rows.append(current_row)
        ordered = []
        for row in rows:
            ordered.extend(sorted(row, key=lambda e: e.bbox.cx))
        return ordered

    return xy_cut(elements)
