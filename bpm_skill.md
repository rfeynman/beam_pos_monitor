##### The BAR BPM.pdf discuss the way to calculate bpm signal. and cited SLAC_bpm.pdf which has more details and BeampositionMonitor.gdf which has boundary details.
Create a python code for the BPM.
* Create a YAML file for the Input parameter:
1. bunch charge and beam charge density , the beam charge density gives out in following two cases is arbitary unit, should be normalized first(intgral along the z=1), then times the bunch charge is the real .two cases 
i) Gaussian distribution, set "sigma" and boundary cut off "n" which is cut off +-n sigma. if n==0, then no cut.
ii)Any density, using array setting different z with the peak current {{z, peakcurrent}...},
2. chamber geometry three cases:
i) round, need input radius
ii) ellipse: need input a, b axis, a is on x direction and b is the y direction
iii) Polygon: provide array of the polygon points list
3. buttons geometry: radius, number of buttons, position
4. filter section parameters, like cutoff frequency
5. initial particles transverse distribution, always use rectangular shape, input x, y, and particles number. particles shall evenly distributed on the x, y rectangular shape as shown in the paper Fig 11 or 12 blue

* output 
1. the Figure 11 and Figure 12, has the input particles. Don't need both a, and b, just one figure for linearity and one figure for nth-order polynomial fit for the geometry defined in YAML
2. The botton using different color with thicker line in the 2D cross section.
3. the figure 13 about the BPM resolution
4. an analysis report summarize the results
