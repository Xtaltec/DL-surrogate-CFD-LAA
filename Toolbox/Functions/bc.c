/*udf file for Boundaries and time steps */


#include "udf.h"
#include "dynamesh_tools.h"

DEFINE_PROFILE(pvs_profile, t, i)        /* Pulmonary veins*/

{
	face_t f;
	real ti = CURRENT_TIME;  	/* total time */
	real j;

     begin_f_loop(f,t)
	 {
		if (ti< 0.400)     /* time SYSTOLE */
			 F_PROFILE(f, t, i)=0.17+((0.18/2)*(1+cos(((2*3.14)/(0.400))*(ti-0.200))));
		else
			if (ti < 0.850)
			F_PROFILE(f, t, i)=0.170+((0.24/2)*(1+cos(((2*3.14)/(0.450-0.0))*(ti-0.625)))); 
	
			else  
			F_PROFILE(f, t, i)=0.170-((0.17/2)*(1+cos(((2*3.14)/(0.650-0.450))*(ti-0.950))));

	 }
	end_f_loop(f,t)
}
