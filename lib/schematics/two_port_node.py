import schemdraw as schem
import schemdraw.elements as e
from schemdraw import dsp
from schemdraw.segments import *
from schemdraw.transform import Transform


class Node2P(e.Element):
    def __init__(
        self,
        name="2-Port\nNode",
        x11="$x_{11}$",
        x12="$x_{12}$",
        x21="$x_{21}$",
        x22="$x_{22}$",
        inp="->",
        outp="<-",
        **kwargs,
    ):
        super().__init__(**kwargs)

        xoff = 2.1

        edgeOff = 0.4
        r = 0.075
        p1a = (xoff - r, 3 - edgeOff)
        p1b = (xoff - r, edgeOff)

        p2a = (xoff + 3 + r, 3 - edgeOff)
        p2b = (xoff + 3 + r, edgeOff)

        # self.segments.append(dsp.Box())
        self.segments.append(
            Segment(
                [
                    (xoff + 0, 0),
                    (xoff + 0, 3),
                    (xoff + 3, 3),
                    (xoff + 3, 0),
                    (xoff + 0, 0),
                ]
            )
        )
        self.segments.append(SegmentCircle(p1a, r))
        self.segments.append(Segment([(p1a[0] - r, p1a[1]), (p1a[0] - 2.0, p1a[1])]))
        self.segments.append(SegmentCircle((p1a[0] - 2.0 - r, p1a[1]), r))

        self.segments.append(SegmentCircle(p1b, r))
        self.segments.append(Segment([(p1b[0] - r, p1b[1]), (p1b[0] - 2.0, p1b[1])]))
        self.segments.append(SegmentCircle((p1b[0] - 2.0 - r, p1b[1]), r))

        self.segments.append(SegmentCircle(p2a, r))
        self.segments.append(Segment([(p2a[0] + r, p2a[1]), (p2a[0] + 2.0, p2a[1])]))
        self.segments.append(SegmentCircle((p2a[0] + 2.0 + r, p2a[1]), r))

        self.segments.append(SegmentCircle(p2b, r))
        self.segments.append(Segment([(p2b[0] + r, p2b[1]), (p2b[0] + 2.0, p2b[1])]))
        self.segments.append(SegmentCircle((p2b[0] + 2.0 + r, p2b[1]), r))

        self.anchors["p1a"] = (p1a[0] - 2.0 - r * 2, p1a[1])
        self.anchors["p1b"] = (p1b[0] - 2.0 - r * 2, p1b[1])
        self.anchors["p2a"] = (p2a[0] + 2.0 + r * 2, p2a[1])
        self.anchors["p2b"] = (p2b[0] + 2.0 + r * 2, p2b[1])

        self.segments.append(SegmentText((xoff + 1.5, 1.5), label=name, color="blue"))
        self.segments.append(SegmentText((xoff + 0.25, 1.5), label="p1"))
        self.segments.append(SegmentText((xoff + 3 - 0.25, 1.5), label="p2"))

        self.segments.append(
            SegmentText((xoff - 1.0 - 2.5, 1.5), label=inp, color="blue", fontsize=12)
        )
        self.segments.append(
            SegmentText((xoff + 4.0 + 2.5, 1.5), label=outp, color="blue", fontsize=12)
        )

        # input
        self.segments.append(
            SegmentArc(
                (xoff - 1.0, 1.5),
                width=1.5,
                height=1.5,
                arrow="cw",
                theta1=-90,
                theta2=100,
                color="blue",
            )
        )
        self.segments.append(
            SegmentText((xoff - 1.0 - 0.5, 1.5), label=x11, color="blue", fontsize=12)
        )

        # ouput
        self.segments.append(
            SegmentArc(
                (xoff + 4.0, 1.5),
                width=1.5,
                height=1.5,
                arrow="ccw",
                theta1=90,
                theta2=-100,
                color="blue",
            )
        )
        self.segments.append(
            SegmentText((xoff + 4.0 + 0.5, 1.5), label=x22, color="blue", fontsize=12)
        )

        # forward
        self.segments.append(
            Segment(
                [(xoff - 1.25, 3.5), (xoff + 4.25, 3.5)],
                color="blue",
                arrow="->",
            )
        )
        self.segments.append(
            SegmentText((xoff + 1.5, 4.0), label=x21, color="blue", fontsize=12)
        )

        # reverse
        self.segments.append(
            Segment(
                [(xoff - 1.25, -0.5), (xoff + 4.25, -0.5)],
                color="blue",
                arrow="<-",
            )
        )
        self.segments.append(
            SegmentText((xoff + 1.5, -1.0), label=x12, color="blue", fontsize=12)
        )
