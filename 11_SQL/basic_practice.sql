select 
	c.email
from
	customer as c
where c.email not like '%@'
	and c.email not like '@%'
	and c.email like '%@%';
	

select 
	distinct p.customer_id,
	p.amount
from
	payment p
where amount =  
	(
		select
			amount 
		from
			payment
		order by
			amount desc
		limit 1
	)
order by p.customer_id ;


select 
	rental_date
from
	rental;


select 
	to_char(r.rental_date, 'yyyy'),
	count(r.rental_id)
from
	rental r
group by
	to_char(r.rental_date, 'yyyy');
	

select 
	to_char(rental_date, 'yyyymm'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyymm')
order by
	to_char(rental_date, 'yyyymm');
	


select
	to_char(rental_date, 'yyyy'), 
	to_char(rental_date, 'mm'),
	count(rental_id)
from
	rental
group by
rollup 
(
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm')
)
order by
to_char(rental_date, 'yyyy'),
to_char(rental_date, 'mm');

-- rollup
select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd'),
	count(rental_id)
from
	rental
group by
rollup 
(
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd')	
)
order by
(
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd')	
);


-- grouping sets
select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd'),
	count(rental_id)
from
	rental
group by
grouping sets
(
	(
		to_char(rental_date, 'yyyy'),
		to_char(rental_date, 'mm'),
		to_char(rental_date, 'dd')
	),
	(
		to_char(rental_date, 'yyyy'),
		to_char(rental_date, 'mm')
	),
	(
		to_char(rental_date, 'yyyy')
	),
	(
	)
)
order by
(
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd')	
);


-- union all
select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd')
union all
select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	null,
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm')
union all
select 
	to_char(rental_date, 'yyyy'),
	null,
	null,
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyy')
union all
select 
	null,
	null,
	null,
	count(rental_id)
from
	rental;
	

select 
	to_char(rental_date, 'dd'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'dd')

	
select 
	null,
	count(rental_id)
from
	rental;

