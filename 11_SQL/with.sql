select 
	film_id,
	title,
	(case
		when length < 30 then 'short'
		when length >=30 and length < 90 then 'medium'
		when length > 90 then 'long'
		end) length
from film;


with tmp1 as (
select 
	film_id,
	title,
	(case
	when length < 30 then 'short'
	when length >=30 and length <90 then'medium'
	when length > 90 then 'long'
	end ) length
from 
	film
)
select * from tmp1;


with  tmp1 as (
select 
	film_id,
	title,
	(case
	when length < 30 then 'short'
	when length >=30 and length < 90 then 'medium'
	when length >90 then 'long'
	end ) length
from 
	film
)
select * from tmp1 where length = 'long' ;