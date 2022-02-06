select 
	sum(
		case 
		when rental_rate = 0.99 then 1
		else 0 end) as "c",
	sum(
		case 
		when rental_rate = 2.99 then 1
		else 0 end) as "b",
	sum(
		case
		when rental_rate = 4.99 then 1
		else 0 end) as "a"
	from film;
		

select 
	rental_rate, count(*) cnt
from
	film
group by rental_rate;


select 
	sum (
			case 
			when rental_rate = 0.99 then 1
			else 0 
			end
		) as "c",
	sum (
			case 
			when rental_rate = 2.99 then 1
			else 0
			end
		) as "b",
	sum (
			case 
			when rental_rate = 4.99 then 1
			else 0
			end
		) as "a"
from film;


select *
from (
		select
			sum(case when rental_rate = 0.99 then cnt else 0 end) as c,
			sum(case when rental_rate = 2.99 then cnt else 0 end) as b,
			sum(case when rental_rate = 4.99 then cnt else 0 end) as a
		from
			(
				select
					rental_rate, count(*) cnt
				from
					film
				group by rental_rate
			) a
		) a
		;