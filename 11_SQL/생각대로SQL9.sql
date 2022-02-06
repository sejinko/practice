-- 문제 1번
select * from rental limit 10;

with TMP1 as
(
select 
	customer_id,
	count(rental_id)
from
	rental r 
group by
	customer_id
order by 
	count(rental_id) desc
)
select
	c.customer_id,
	c.first_name,
	c.last_name
from
	tmp1 t
	join customer c on t.customer_id = c.customer_id 
limit 1;


-- 문제 2번
select * from film limit 10;

/*
 x <= 60 : short
 x < 60 <= 120 : middle
 x > 120 : long
*/

select 
	case
		when length <= 60 then 'short'
		when length > 60 and length <=120 then 'middle'
		when length > 120 then 'long'
	end as length_type,
	count(case
		when length <= 60 then 'short'
		when length > 60 and length <=120 then 'middle'
		when length > 120 then 'long'
	end)
from 
	film f
group by
	case
		when length <= 60 then 'short'
		when length > 60 and length <=120 then 'middle'
		when length > 120 then 'long'
	end;
	

-- 문제 3번
select * from film limit 10;


with TM1 as
(
select 
	film_id,
	case 
		when f.rating = 'G' then 'general Audiences'
		when f.rating ='PG' then 'Parental Guidance Suggested.'
		when f.rating ='PG-13' then 'Parents Strongly Cautioned'
		when f.rating ='R' then 'Restricted'
		when f.rating ='NC-17' then 'then no One 17 and under Admitted'
	end as 영문명,
	case 
		when f.rating ='G' then '모든 연령대 시청가능'
		when f.rating ='PG' then '모든 연령대 시청가능하나, 부모의 지도가 필요'
		when f.rating ='PG-13' then '13세 미만의 아동에게 부적절 할 수 있으며, 부모의 주의를 요함'
		when f.rating ='R' then '17세 또는 그이상의 성인'
		when f.rating ='NC-17' then '17세 이하 시청 불가'
	end as 한글명
from
	film f
)
select
	영문명 ||' ('|| 한글명 || ')' as full_name
from 
	TM1;


-- 문제 4	번
select * from rental limit 10;

select
	customer_id,
	count(rental_id),
	case 
		when count(rental_id) >= 31 then 'A'
		when count(rental_id) >=21 and count(rental_id) < 31 then 'B'
		when count(rental_id) >= 11 and count(rental_id) < 20 then 'C'
		when count(rental_id) < 10 then 'D'
	end as grade
from 
	rental r 
group by
	customer_id;
	

-- 문제 5번
select * from customer limit 10;

select 
	first_name,
	(case 
		when first_name like 'A@' then 'A'
		when first_name like 'B@' then 'B'
		when first_name like 'C@' then 'C'
		else 'Other'
	end) as flag
from
	customer;
--???


-- 문제 6번
select * from payment limit 10;



	
with TMP1 as
(
select 
	customer_id,
	staff_id,
	to_char(p.payment_date, 'yyyymm') as p_date
from
	payment p
where
	to_char(p.payment_date, 'yyyymm') >= '200701'
and	to_char(p.payment_date, 'yyyymm') <= '200703'
)	
select 
	customer_id,
	(case 
		when staff_id = 2 then 'Y'
		else 'N'
	end) as staff_2
from
	TMP1;
	

-- 7번
select * from payment limit 10;

with TMP1 as
(
select 
	customer_id,
	payment_date,
	to_char(payment_date, 'mm') as mm
from
	payment
)
select
	customer_id, 
	payment_date,
	(case
		when mm >= '01' and mm < '03' then 'Q1'
		when mm >= '04' and mm < '06' then 'Q2'
		when mm >= '07' and mm < '09' then 'Q3'
		when mm >= '10' and mm < '12' then 'Q12'
	end) as Q
from 
	TMP1;
	

-- 문제 8번
select * from rental limit 10;



-- 문제 9번
select * from rental limit 10;

select 
	staff_id,
	return_date,
	count(rental_id) over(partition by staff_id) as flag
	(case 
		when count(rental_id) over(partition by staff_id) >= 0 and count(rental_id) over(partition by staff_id) < 500 then 'under_500'
		when count(rental_id) over(partition by staff_id) >= 500 and count(rental_id) over(partition by staff_id) < 3000 then 'under_3000'
		when count(rental_id) over(partition by staff_id) >= 3000 and count(rental_id) over(partition by staff_id) < 99999 then 'over_3001'
	end) as dd
from
	rental;
	

-- 문제 10번
select * from staff limit 10;


with TMP1 as
(
select 
	staff_id,
	password as old_pw
	
from
	staff s 
)
select
