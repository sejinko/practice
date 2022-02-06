-- 문제 1번
select * from rental limit 10;
select * from customer limit 10;

select 
	c.customer_id,
	c.first_name,
	c.last_name,
	sum(r.rental_id)
from
	rental r
	join customer c on r.customer_id = c.customer_id
group by
	r.rental_id
order by 




-- 문제 5
select * from payment limit 10;

select 
	to_char(payment_date, 'yyyymm') as month,
	sum(amount)
from
	payment p 
group by
	to_char(payment_date, 'yyyymm') 
having 


-- 문제 6
select * from rental limit 10;
select * from city limit 10;
select * from customer limit 10;
select * from payment limit 10;

select 
	city,
	dense_rank()
	rank()
	