select *
from customer
;


select
	first_name
	,last_name
	,email
from
	customer
;

-- order by
select 
	first_name
	,last_name
from
	customer
order by first_name
;

select 
	first_name
	,last_name
from
	customer
order by 2 asc 
		,1 desc
;



