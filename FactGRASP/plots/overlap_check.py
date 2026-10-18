# checks that no text in a figure overlaps before it is saved

def check(fig, name="fig"):
    fig.canvas.draw()
    ann, ticks = [], []
    for ax in fig.axes:
        ann += list(ax.texts)
        if ax.get_legend() is not None:
            ann.append(ax.get_legend())
        if ax.title.get_text().strip():
            ann.append(ax.title)
        if not ax.axison:
            continue
        ticks += [t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text().strip()]
    ann += list(fig.texts)
    def bb(o):
        try:
            return o.get_window_extent()
        except Exception:
            return None
    def hit(a, b):
        ba, bb_ = bb(a), bb(b)
        if ba is None or bb_ is None:
            return False
        return not (ba.x1 < bb_.x0 or bb_.x1 < ba.x0 or ba.y1 < bb_.y0 or bb_.y1 < ba.y0)
    n = 0
    for i in range(len(ann)):
        for j in range(i + 1, len(ann)):
            if hit(ann[i], ann[j]):
                n += 1
    for a in ann:
        for t in ticks:
            if hit(a, t):
                n += 1
    print(f"[{name}] text/tick overlaps = {n}")
    return n
