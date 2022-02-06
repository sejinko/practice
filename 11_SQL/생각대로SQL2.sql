-- ���� 1��
select * from film;
select 
	*
from 
	film
limit 100;


-- ���� 2��
select * from actor;
select 
	actor_id,
	last_name,
	first_name
from
	actor
where 
	last_name like 'Jo%'  -- �빮��, �ҹ��� ������
order by
	actor_id
limit 1;


-- ���� 3��
select * from country limit 10;
select 
	*
from 
	country
where 
	country_id between 2 and 9;


-- ���� 4��
select * from country limit 10;
select 
	country
from
	country
where 
	country like 'A%';


-- ���� 5��
select 
	country
from
	country
where
	country like '%s';


-- ���� 6��
select * from address limit 10;
select 
	address_id,
	address,
	district,
	postal_code
from
	address
where
	postal_code like '77%';


-- ���� 7��
select
	address_id,
	address,
	district,
	postal_code
from
	address
where 
	postal_code like '_1%';


-- ���� 8��
select * from payment limit 10;
select 
	*
from 
	payment
where 
	customer_id = 341
and to_char(payment_date, 'yyyymmdd') between '20070215' and '20070216';


-- ���� 9��
  -- ���� 1
select
	*
from 
	payment
where 
	customer_id = 355
and amount between 1 and 3;
  -- ���� 2
select 
	*
from 
	payment
where 
	customer_id = 355
and amount > 1
and amount < 3;


-- ���� 10��
select * from customer limit 10;

select 
	customer_id,
	last_name,
	first_name
from 
	customer c 
where 
	first_name in ('Maria', 'Lisa', 'Mike');


-- ���� 11��
select * from film limit 10;
select
	*
from
	film f 
where 
	(100 <= length and length <= 120)
or  (3 <= rental_duration and rental_duration <= 5);


select
	*
from
	film f 
where 
	length between 100 and 120
or  rental_duration between 3 and 5;


	
-- ���� 12��
select * from address limit 10 ;
select 
	*
from 
	address a 
where 
	postal_code in ('', '45200', '17886');


-- ���� 13��
select * from address limit 10;
select 
	*
from 
	address a 
where 
	address2 isnull;  -- isnull �׳� �̷��� ���� �ǳ�?
	
	
-- ���� 14��
select * from staff limit 10;
select 
	staff_id,
	first_name || ' ' || last_name as name
from 
	staff s 
where 
	picture notnull;  -- notnull �̷��� ���� �ǳ�?
	
	
-- ���� 15��
select * from rental limit 10;
select 
	*
from 
	rental r 
where
	return_date isnull;

select 
	count(*)
from 
	rental r 
where
	return_date isnull;


-- ���� 16��
select * from address limit 10;
select 
	*
from 
	address a 
where 
	postal_code in ('', '35200', '17886');  -- null�̸� ������ ����? null�̶�� �����ִ� ���� ���°�?

	
-- ���� 17��
select * from customer limit 10;
select
	*
from 
	customer c 
where 
	last_name like '%John%';  -- %�� �־ �ǰ� ��� �Ǵ� ����


-- ���� 18��
select * from address limit 10;
select
	*
from 
	address a
where 
	address2 is null;


